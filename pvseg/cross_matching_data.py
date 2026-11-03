"""Geographically separated support-query episodes from PV-896."""

import random
from collections import Counter

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from pvseg.data import PV896Dataset
from pvseg.spatial import scene_groups


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
POSITIVE_CATEGORIES = ('正样本', '困难正样本')


def spatial_partition(rows, seed=42, validation_fraction=0.2):
    groups, overlap_edges = scene_groups(rows)
    if len(groups) < 3 or not 0 < validation_fraction < 1:
        raise ValueError('Need at least three spatial groups and a valid holdout fraction')
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    validation_count = max(1, round(len(keys) * validation_fraction))
    validation_groups = set(keys[:validation_count])
    scene_group = {scene: group for group, scenes in groups.items() for scene in scenes}
    train = [row for row in rows if scene_group[row['source_scene']] not in validation_groups]
    validation = [row for row in rows if scene_group[row['source_scene']] in validation_groups]
    if not train or not validation or any(
            not any(row['category'] in POSITIVE_CATEGORIES for row in partition)
            for partition in (train, validation)):
        raise ValueError('Training and validation both need photovoltaic samples')
    report = {
        'seed': seed, 'validation_fraction': validation_fraction,
        'validation_groups': sorted(validation_groups), 'spatial_groups': groups,
        'overlap_edges': overlap_edges, 'train_counts': dict(Counter(row['category'] for row in train)),
        'validation_counts': dict(Counter(row['category'] for row in validation)),
        'train_ids': [row['sample_id'] for row in train],
        'validation_ids': [row['sample_id'] for row in validation],
        'note': 'Development holdout from supplied geographic metadata; not an independent test set.',
    }
    return train, validation, scene_group, report


def crop_origin(mask, crop_size, rng, require_foreground=False):
    if mask.ndim != 3 or mask.shape[0] != 1:
        raise ValueError('Expected a 1,H,W mask')
    height, width = mask.shape[-2:]
    if crop_size > min(height, width):
        raise ValueError('Crop exceeds source image')
    if require_foreground:
        coordinates = torch.nonzero(mask[0] > 0, as_tuple=False)
        if len(coordinates) == 0:
            raise ValueError('Cannot make positive crop from an empty mask')
        center_row, center_column = coordinates[rng.randrange(len(coordinates))].tolist()
        row_min = max(0, center_row - crop_size + 1)
        row_max = min(center_row, height - crop_size)
        column_min = max(0, center_column - crop_size + 1)
        column_max = min(center_column, width - crop_size)
        return rng.randint(row_min, row_max), rng.randint(column_min, column_max)
    return rng.randint(0, height - crop_size), rng.randint(0, width - crop_size)


def tile_origins(image_size, tile_size):
    if image_size < tile_size or tile_size <= 0:
        raise ValueError('Tile must fit inside image')
    starts = list(range(0, image_size - tile_size + 1, tile_size))
    if starts[-1] != image_size - tile_size:
        starts.append(image_size - tile_size)
    return [(row, column) for row in starts for column in starts]


class PVEpisodes(Dataset):
    def __init__(self, root, query_rows, support_rows, scene_group, shots=1,
                 crop_size=256, seed=42, training=True):
        if shots < 1 or crop_size < 64 or crop_size > 896 or crop_size % 32:
            raise ValueError('Shots must be positive and crop size must be 64..896, divisible by 32')
        self.query_rows = list(query_rows)
        self.support_rows = [row for row in support_rows if row['category'] in POSITIVE_CATEGORIES]
        self.scene_group = scene_group
        self.shots, self.crop_size, self.seed, self.training = shots, crop_size, seed, training
        all_rows = {row['sample_id']: row for row in self.query_rows + self.support_rows}
        self.rows = list(all_rows.values())
        self.images = PV896Dataset(root, self.rows, IMAGENET_MEAN, IMAGENET_STD,
                                   img_size=896, augment=False)
        self.by_id = {row['sample_id']: index for index, row in enumerate(self.rows)}
        self.support_groups = {}
        for row in self.support_rows:
            self.support_groups.setdefault(scene_group[row['source_scene']], []).append(row)
        for query in self.query_rows:
            available = len(self._available_groups(query))
            if available < shots:
                raise ValueError(f"{query['sample_id']} has only {available} independent support groups")

    def __len__(self):
        return len(self.query_rows)

    def _available_groups(self, query):
        query_group = self.scene_group[query['source_scene']]
        return [group for group in sorted(self.support_groups) if group != query_group]

    def support_for(self, query, rng):
        groups = rng.sample(self._available_groups(query), self.shots)
        return [rng.choice(self.support_groups[group]) for group in groups]

    def load(self, row):
        item = self.images[self.by_id[row['sample_id']]]
        return item['image'], item['mask']

    def support_patches(self, query, rng):
        images, masks, ids = [], [], []
        for selected in self.support_for(query, rng):
            group = self.scene_group[selected['source_scene']]
            alternatives = [row for row in rng.sample(self.support_groups[group],
                                                       len(self.support_groups[group]))
                            if row['sample_id'] != selected['sample_id']]
            for row in [selected] + alternatives:
                image, mask = self.load(row)
                for _ in range(32):
                    top, left = crop_origin(mask, self.crop_size, rng, require_foreground=True)
                    patch_mask = mask[:, top:top + self.crop_size, left:left + self.crop_size]
                    grid_mask = F.interpolate(patch_mask[None], size=(self.crop_size // 8,) * 2,
                                              mode='bilinear', align_corners=True)
                    if grid_mask.any():
                        break
                else:
                    continue
                images.append(image[:, top:top + self.crop_size, left:left + self.crop_size])
                masks.append(patch_mask)
                ids.append(row['sample_id'])
                break
            else:
                raise ValueError(f'No valid stride-eight support crop in group {group}')
        return torch.stack(images), torch.stack(masks), ids

    def __getitem__(self, index):
        query = self.query_rows[index]
        rng = random if self.training else random.Random(self.seed + index)
        image, mask = self.load(query)
        has_foreground = query['category'] in POSITIVE_CATEGORIES
        top, left = crop_origin(mask, self.crop_size, rng, require_foreground=has_foreground)
        query_patch = image[:, top:top + self.crop_size, left:left + self.crop_size]
        target = mask[:, top:top + self.crop_size, left:left + self.crop_size]
        support_images, support_masks, support_ids = self.support_patches(query, rng)
        if self.training:
            rotations = rng.randrange(4)
            query_patch, target = torch.rot90(query_patch, rotations, (-2, -1)), torch.rot90(
                target, rotations, (-2, -1))
            if rng.random() < 0.5:
                query_patch, target = query_patch.flip(-1), target.flip(-1)
        return {'query': query_patch, 'target': target.long().squeeze(0),
                'supports': support_images, 'support_masks': support_masks,
                'query_id': query['sample_id'], 'support_ids': support_ids,
                'category': query['category']}
