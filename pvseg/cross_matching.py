"""ResNet50 cross-matching baseline adapted from the SCCNet architecture."""

import time

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50


class ResNet50Features(nn.Module):
    """Return pre-activation bottleneck maps from layers 4, 3, and 2."""

    def __init__(self):
        super().__init__()
        self.backbone = resnet50(weights=None)
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    @staticmethod
    def _bottleneck(block, features):
        residual = features if block.downsample is None else block.downsample(features)
        output = block.relu(block.bn1(block.conv1(features)))
        output = block.relu(block.bn2(block.conv2(output)))
        output = block.bn3(block.conv3(output)) + residual
        return output, F.relu(output)

    def forward(self, images):
        self.backbone.eval()
        with torch.no_grad():
            features = self.backbone.maxpool(self.backbone.relu(
                self.backbone.bn1(self.backbone.conv1(images))))
            groups = []
            for layer in (self.backbone.layer1, self.backbone.layer2,
                          self.backbone.layer3, self.backbone.layer4):
                maps = []
                for block in layer:
                    pre_activation, features = self._bottleneck(block, features)
                    maps.append(pre_activation)
                groups.append(maps)
        return [groups[3], groups[2], groups[1]]


def cross_correlations(query_groups, support_groups, support_mask):
    if len(query_groups) != 3 or len(support_groups) != 3:
        raise ValueError('Expected three ResNet feature groups')
    if support_mask.ndim != 4 or support_mask.shape[1] != 1:
        raise ValueError('Support mask must have shape B,1,H,W')
    if not torch.isfinite(support_mask).all() or (support_mask < 0).any() or (support_mask > 1).any():
        raise ValueError('Support mask must contain finite values in [0,1]')
    if not (support_mask.flatten(1).sum(1) > 0).all():
        raise ValueError('Each support mask needs photovoltaic foreground')

    pyramid = []
    for group_index, (query_maps, support_maps) in enumerate(zip(query_groups, support_groups)):
        if len(query_maps) != len(support_maps) or not query_maps:
            raise ValueError('Query/support feature groups differ')
        layer_correlations = []
        for query_map, support_map in zip(query_maps, support_maps):
            if query_map.shape[:2] != support_map.shape[:2]:
                raise ValueError('Query/support feature channels differ')
            support_weight = F.interpolate(support_mask.float(), size=support_map.shape[-2:],
                                           mode='bilinear', align_corners=True)
            if group_index == 2 and not (support_weight.flatten(1).sum(1) > 0).all():
                raise ValueError('Support foreground vanished at the finest feature resolution')
            masked_support = support_map * support_weight
            batch, channels, query_height, query_width = query_map.shape
            support_height, support_width = support_map.shape[-2:]
            query_vectors = query_map.reshape(batch, channels, -1)
            support_vectors = masked_support.reshape(batch, channels, -1)
            query_vectors = query_vectors / (query_vectors.norm(dim=1, keepdim=True) + 1e-5)
            support_vectors = support_vectors / (support_vectors.norm(dim=1, keepdim=True) + 1e-5)
            correlation = torch.bmm(query_vectors.transpose(1, 2), support_vectors)
            layer_correlations.append(correlation.clamp_min(0).reshape(
                batch, query_height, query_width, support_height, support_width))
        pyramid.append(torch.stack(layer_correlations, dim=1))
    return pyramid


class CenterPivotConv4d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel, support_stride):
        super().__init__()
        self.support_stride = support_stride
        self.query_conv = nn.Conv2d(input_channels, output_channels, kernel,
                                    padding=kernel // 2)
        self.support_conv = nn.Conv2d(input_channels, output_channels, kernel,
                                      padding=kernel // 2, stride=support_stride)

    def forward(self, tensor):
        batch, channels, query_height, query_width, support_height, support_width = tensor.shape
        sampled = tensor[..., ::self.support_stride, ::self.support_stride]
        sampled_height, sampled_width = sampled.shape[-2:]
        query_input = sampled.permute(0, 4, 5, 1, 2, 3).reshape(
            batch * sampled_height * sampled_width, channels, query_height, query_width)
        query_output = self.query_conv(query_input)
        output_channels, output_height, output_width = query_output.shape[-3:]
        query_output = query_output.reshape(batch, sampled_height, sampled_width,
                                            output_channels, output_height, output_width)
        query_output = query_output.permute(0, 3, 4, 5, 1, 2)

        support_input = tensor.permute(0, 2, 3, 1, 4, 5).reshape(
            batch * query_height * query_width, channels, support_height, support_width)
        support_output = self.support_conv(support_input)
        support_output = support_output.reshape(
            batch, query_height, query_width, output_channels,
            support_output.shape[-2], support_output.shape[-1])
        support_output = support_output.permute(0, 3, 1, 2, 4, 5)
        return query_output + support_output


def _squeeze_block(input_channels, channels, kernels, strides):
    layers = []
    for output_channels, kernel, stride in zip(channels, kernels, strides):
        layers.extend((CenterPivotConv4d(input_channels, output_channels, kernel, stride),
                       nn.GroupNorm(4, output_channels), nn.ReLU(inplace=True)))
        input_channels = output_channels
    return nn.Sequential(*layers)


class HypercorrelationSqueeze(nn.Module):
    """SCCNet cross-matching branch: 4D squeeze, pyramid fusion, 2D decoder."""

    def __init__(self, group_channels=(3, 6, 4)):
        super().__init__()
        self.deep = _squeeze_block(group_channels[0], (16, 64, 128), (3, 3, 3), (2, 2, 2))
        self.middle = _squeeze_block(group_channels[1], (16, 64, 128), (5, 3, 3), (4, 2, 2))
        self.shallow = _squeeze_block(group_channels[2], (16, 64, 128), (5, 5, 3), (4, 4, 2))
        self.merge_deep_middle = _squeeze_block(128, (128, 128, 128), (3, 3, 3), (1, 1, 1))
        self.merge_middle_shallow = _squeeze_block(128, (128, 128, 128), (3, 3, 3), (1, 1, 1))
        self.decoder = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
                                     nn.Conv2d(128, 64, 3, padding=1), nn.ReLU())
        self.classifier = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
                                        nn.Conv2d(64, 2, 3, padding=1))

    @staticmethod
    def _align_query(tensor, spatial_size):
        batch, channels, query_height, query_width, support_height, support_width = tensor.shape
        reshaped = tensor.permute(0, 4, 5, 1, 2, 3).reshape(
            batch * support_height * support_width, channels, query_height, query_width)
        resized = F.interpolate(reshaped, size=spatial_size, mode='bilinear', align_corners=True)
        return resized.reshape(batch, support_height, support_width, channels, *spatial_size).permute(
            0, 3, 4, 5, 1, 2).contiguous()

    def forward(self, pyramid):
        if len(pyramid) != 3:
            raise ValueError('Expected deep, middle, and shallow hypercorrelations')
        deep, middle, shallow = self.deep(pyramid[0]), self.middle(pyramid[1]), self.shallow(pyramid[2])
        merged = self.merge_deep_middle(self._align_query(deep, middle.shape[-4:-2]) + middle)
        merged = self.merge_middle_shallow(self._align_query(merged, shallow.shape[-4:-2]) + shallow)
        flattened = merged.mean(dim=(-2, -1))
        decoded = self.decoder(flattened)
        doubled_size = (decoded.shape[-2] * 2, decoded.shape[-1] * 2)
        return self.classifier(F.interpolate(decoded, size=doubled_size,
                                             mode='bilinear', align_corners=True))


class CrossMatchingNet(nn.Module):
    def __init__(self, encoder=None, group_channels=(3, 6, 4)):
        super().__init__()
        self.encoder = encoder if encoder is not None else ResNet50Features()
        self.matcher = HypercorrelationSqueeze(group_channels)
        self.last_forward_profile = None
        self.last_nshot_profiles = None

    def forward(self, query_image, support_image, support_mask, profile=False):
        self.last_forward_profile = None
        if query_image.ndim != 4 or support_image.ndim != 4 or query_image.shape != support_image.shape:
            raise ValueError('Query and support images must have matching B,3,H,W shapes')
        if query_image.shape[1] != 3 or min(query_image.shape[-2:]) < 64:
            raise ValueError('Expected RGB images at least 64 pixels high and wide')
        if support_mask.shape != (support_image.shape[0], 1, *support_image.shape[-2:]):
            raise ValueError('Support mask size differs from support image')
        device = query_image.device
        if profile:
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
        query_groups = self.encoder(query_image)
        support_groups = self.encoder(support_image)
        pyramid = cross_correlations(query_groups, support_groups, support_mask)
        logits = self.matcher(pyramid)
        output = F.interpolate(logits, size=query_image.shape[-2:],
                               mode='bilinear', align_corners=True)
        if profile:
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            self.last_forward_profile = {
                'query_feature_shapes': [[list(feature.shape) for feature in group]
                                         for group in query_groups],
                'support_feature_shapes': [[list(feature.shape) for feature in group]
                                           for group in support_groups],
                'correlation_shapes': [list(correlation.shape) for correlation in pyramid],
                'correlation_elements': [correlation.numel() for correlation in pyramid],
                'logits_shape': list(output.shape),
                'forward_seconds': time.perf_counter() - started,
                'forward_peak_allocated_bytes': torch.cuda.max_memory_allocated(device)
                if device.type == 'cuda' else None,
            }
        return output

    @torch.no_grad()
    def predict_nshot(self, query_image, support_images, support_masks, vote_threshold=0.4,
                      profile=False):
        if support_images.ndim != 5 or support_masks.ndim != 5:
            raise ValueError('Support tensors must have B,K,C,H,W shapes')
        if support_images.shape[:2] != support_masks.shape[:2] or support_images.shape[1] < 1:
            raise ValueError('Support image/mask shot counts differ')
        if not 0 < vote_threshold <= 1:
            raise ValueError('Vote threshold must be in (0,1]')
        predictions = []
        profiles = []
        for shot in range(support_images.shape[1]):
            logits = self(query_image, support_images[:, shot], support_masks[:, shot],
                          profile=profile)
            predictions.append(logits.argmax(dim=1))
            if profile:
                profiles.append(self.last_forward_profile)
        self.last_nshot_profiles = profiles if profile else None
        if len(predictions) == 1:
            return predictions[0]
        votes = torch.stack(predictions).sum(dim=0)
        peak = votes.flatten(1).amax(dim=1).clamp_min(1).reshape(-1, 1, 1)
        return (votes.float() / peak >= vote_threshold).long()
