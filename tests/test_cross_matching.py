"""Synthetic checks for the SCCNet Cross-Matching-only baseline."""

import random
import unittest

import torch

from pvseg.cross_matching import CrossMatchingNet, ResNet50Features, cross_correlations
from pvseg.cross_matching_data import crop_origin, tile_origins


class TinyFeatures(torch.nn.Module):
    def forward(self, image):
        base = image.mean(dim=1, keepdim=True)
        return [[torch.nn.functional.avg_pool2d(base, 32)] * 3,
                [torch.nn.functional.avg_pool2d(base, 16)] * 6,
                [torch.nn.functional.avg_pool2d(base, 8)] * 4]


class CrossMatchingChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_resnet_feature_groups_and_frozen_state(self):
        encoder = ResNet50Features().train()
        self.assertFalse(encoder.backbone.training)
        self.assertFalse(any(parameter.requires_grad for parameter in encoder.parameters()))
        maps = encoder(torch.rand(1, 3, 64, 64))
        self.assertEqual([len(group) for group in maps], [3, 6, 4])
        self.assertEqual([group[0].shape[-2:] for group in maps],
                         [torch.Size([2, 2]), torch.Size([4, 4]), torch.Size([8, 8])])

    def test_support_mask_controls_correlations(self):
        query = torch.ones(1, 1, 2, 2)
        support = torch.ones_like(query)
        mask = torch.zeros(1, 1, 2, 2)
        mask[:, :, 0, 0] = 1
        pyramid = cross_correlations([[query] * 3, [query] * 6, [query] * 4],
                                     [[support] * 3, [support] * 6, [support] * 4], mask)
        self.assertEqual([tensor.shape[1] for tensor in pyramid], [3, 6, 4])
        self.assertGreater(pyramid[0][0, 0, 0, 0, 0, 0].item(), 0)
        self.assertEqual(pyramid[0][0, 0, 0, 0, 1, 1].item(), 0)
        with self.assertRaisesRegex(ValueError, 'foreground'):
            cross_correlations([[query] * 3, [query] * 6, [query] * 4],
                               [[support] * 3, [support] * 6, [support] * 4],
                               torch.zeros_like(mask))

    def test_training_step_and_inference_do_not_need_query_mask(self):
        model = CrossMatchingNet(encoder=TinyFeatures())
        query = torch.rand(1, 3, 64, 64)
        support = torch.rand(1, 3, 64, 64)
        mask = torch.zeros(1, 1, 64, 64)
        mask[:, :, 16:48, 16:48] = 1
        logits = model(query, support, mask, profile=True)
        self.assertEqual(tuple(logits.shape), (1, 2, 64, 64))
        self.assertEqual(model.last_forward_profile['correlation_shapes'],
                         [[1, 3, 2, 2, 2, 2], [1, 6, 4, 4, 4, 4],
                          [1, 4, 8, 8, 8, 8]])
        self.assertGreater(model.last_forward_profile['forward_seconds'], 0)
        loss = torch.nn.functional.cross_entropy(logits, mask[:, 0].long())
        loss.backward()
        self.assertTrue(any(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                            for parameter in model.matcher.parameters()))
        model.eval()
        prediction = model.predict_nshot(query, support[:, None].repeat(1, 2, 1, 1, 1),
                                        mask[:, None].repeat(1, 2, 1, 1, 1))
        self.assertEqual(tuple(prediction.shape), (1, 64, 64))
        self.assertTrue(torch.isin(prediction, torch.tensor([0, 1])).all())

    def test_native_resolution_tiling_and_positive_support_crop(self):
        origins = tile_origins(896, 256)
        self.assertEqual(len(origins), 16)
        covered = torch.zeros(896, 896, dtype=torch.bool)
        for top, left in origins:
            covered[top:top + 256, left:left + 256] = True
        self.assertTrue(covered.all())
        mask = torch.zeros(1, 896, 896)
        mask[:, 700, 800] = 1
        top, left = crop_origin(mask, 256, random.Random(42),
                                require_foreground=True)
        self.assertEqual(mask[:, top:top + 256, left:left + 256].sum().item(), 1)


if __name__ == '__main__':
    unittest.main()
