"""896px source validation, resized RGB, and geographically isolated episodes."""

import hashlib
import json
import random

import torch
from torch.utils.data import Dataset

from pvseg.data import CATEGORIES, PV896Dataset
from pvseg.spatial import bounds, overlap, spatial_pilot_split
from .modules import resize


def make_split(rows, pilot=False, seed=42):
    train, validation, report = spatial_pilot_split(rows, CATEGORIES, seed)
    if not pilot:
        assignment = report['scene_assignment_before_sampling']
        train = [r for r in rows if assignment[r['source_scene']] == 'train']
        validation = [r for r in rows if assignment[r['source_scene']] == 'validation']
    report.update(train_ids=[r['sample_id'] for r in train],
                  validation_ids=[r['sample_id'] for r in validation], pilot=pilot,
                  evaluation_scope='development validation; all source scenes already used during prior development')
    encoded = json.dumps({'train': report['train_ids'], 'validation': report['validation_ids']}, sort_keys=True)
    report['split_sha256'] = hashlib.sha256(encoded.encode()).hexdigest()
    return train, validation, report


class ResizedPV(Dataset):
    def __init__(self, root, rows, size=512, augment=False):
        if size < 32 or size % 16:
            raise ValueError('Model size must be a multiple of 16 and >=32')
        self.raw = PV896Dataset(root, rows, [0, 0, 0], [1, 1, 1], img_size=896, augment=augment)
        self.rows, self.size = list(rows), size

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, index):
        item = self.raw[index]
        item['image'] = resize(item['image'][None], (self.size, self.size))[0].clamp(0, 1)
        item['mask'] = resize(item['mask'][None], (self.size, self.size), 'nearest')[0]
        return item


def eligible_support(query, support, group_for):
    return (CATEGORIES[support['category']] != 'negative' and
            query['sample_id'] != support['sample_id'] and
            group_for[query['source_scene']] != group_for[support['source_scene']] and
            not overlap(bounds(query), bounds(support)) and
            not (query.get('image_sha256') and query.get('image_sha256') == support.get('image_sha256')))


class Episodes(Dataset):
    def __init__(self, queries, supports, groups, shots=1, seed=42):
        self.queries, self.supports, self.shots, self.seed = queries, supports, shots, seed
        self.epoch = 0
        self.group_for = {scene: key for key, scenes in groups.items() for scene in scenes}
        self.candidates = [[i for i, s in enumerate(supports.rows) if eligible_support(q, s, self.group_for)]
                           for q in queries.rows]
        if any(len(c) < shots for c in self.candidates):
            raise ValueError('Too few geographically separate positive support candidates')

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, index):
        query = self.queries[index]
        rng = random.Random(self.seed + 100003 * self.epoch + index)
        candidates = list(self.candidates[index])
        rng.shuffle(candidates)
        selected, selected_rows = [], []
        for candidate in candidates:
            row = self.supports.rows[candidate]
            if any(overlap(bounds(row), bounds(s)) or
                   (row.get('image_sha256') and row.get('image_sha256') == s.get('image_sha256'))
                   for s in selected_rows):
                continue
            item = self.supports[candidate]
            # Reject disappearing targets; do not enlarge masks or inject positive pixels.
            patch_mask = resize(item['mask'][None], (self.supports.size // 16,) * 2, 'nearest')
            if not patch_mask.any():
                continue
            selected.append(item)
            selected_rows.append(row)
            if len(selected) == self.shots:
                break
        if len(selected) != self.shots:
            raise ValueError(f'No {self.shots} nonoverlapping supports surviving downsampling for {query["sample_id"]}')
        return {'query': query['image'], 'target': query['mask'],
                'supports': torch.stack([s['image'] for s in selected]),
                'support_masks': torch.stack([s['mask'] for s in selected]),
                'query_id': query['sample_id'], 'category': query['category'],
                'support_ids': '|'.join(s['sample_id'] for s in selected)}
