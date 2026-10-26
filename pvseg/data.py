"""ID-based PV-896 pairing and strict binary-mask loading."""

import csv
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


CATEGORIES = {
    "\u6b63\u6837\u672c": "positive",
    "\u8d1f\u6837\u672c": "negative",
    "\u56f0\u96be\u6b63\u6837\u672c": "hard_positive",
}


def discover_root(input_root):
    candidates = [p.parent for p in Path(input_root).rglob('manifest.csv')
                  if all((p.parent / c / 'images').is_dir() and
                         (p.parent / c / 'labels').is_dir() for c in CATEGORIES)]
    if len(candidates) != 1:
        raise ValueError(f'Expected one PV-896 root, found {candidates}')
    return candidates[0]


def read_manifest(root):
    root = Path(root).resolve()
    with (root / 'manifest.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    seen = set()
    for row in rows:
        sid = row['sample_id']
        if sid in seen or row['category'] not in CATEGORIES:
            raise ValueError(f'Duplicate ID or unsupported category: {sid}')
        seen.add(sid)
        for key, folder in [('image', 'images'), ('label', 'labels')]:
            path = (root / row[key]).resolve()
            if (not path.is_relative_to(root) or not path.is_file() or path.stem != sid
                    or path.parent.name != folder or path.parent.parent.name != row['category']):
                raise ValueError(f'Invalid manifest ID/path: {sid}, {key}')
    return rows


def select_audit_rows(rows, seed=42):
    rng = random.Random(seed)
    chosen = []
    for category in CATEGORIES:
        pool = sorted((r for r in rows if r['category'] == category), key=lambda r: r['sample_id'])
        chosen.extend(sorted(rng.sample(pool, min(4, len(pool))), key=lambda r: r['sample_id']))
    return chosen


def synchronized_transform(image, mask, horizontal=False, vertical=False, rotations=0):
    if horizontal:
        image, mask = np.flip(image, 1), np.flip(mask, 1)
    if vertical:
        image, mask = np.flip(image, 0), np.flip(mask, 0)
    return (np.ascontiguousarray(np.rot90(image, rotations)),
            np.ascontiguousarray(np.rot90(mask, rotations)))


class PV896Dataset(Dataset):
    def __init__(self, root, rows, mean, std, img_size=896, augment=False,
                 image_dir=None, mask_dir=None):
        self.root, self.rows = Path(root), list(rows)
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        if self.mean.numel() != 3 or self.std.numel() != 3 or (self.std <= 0).any():
            raise ValueError('Expected three verified RGB normalization values')
        if bool(image_dir) != bool(mask_dir):
            raise ValueError('image_dir and mask_dir must be provided together')
        self.image_dir = Path(image_dir) if image_dir else None
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.img_size, self.augment = img_size, augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image_path = self.image_dir / f"{row['sample_id']}.bmp" if self.image_dir else self.root / row['image']
        mask_path = self.mask_dir / f"{row['sample_id']}.bmp" if self.mask_dir else self.root / row['label']
        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            image.load()
            mask.load()
            if image.mode != 'RGB' or mask.mode != 'L':
                raise ValueError(f"Unexpected raw modes for {row['sample_id']}: {image.mode}/{mask.mode}")
            rgb, label = np.array(image), np.array(mask)
        if rgb.shape != (self.img_size, self.img_size, 3) or label.shape != rgb.shape[:2]:
            raise ValueError(f"Unexpected dimensions for {row['sample_id']}: {rgb.shape}/{label.shape}")
        if rgb.dtype != np.uint8 or label.dtype != np.uint8 or not np.isin(label, [0, 255]).all():
            raise ValueError(f"Unexpected dtype or mask encoding for {row['sample_id']}; no nonzero merge")
        target = (label == 255).astype(np.float32)
        fraction = float(target.mean())
        category = CATEGORIES[row['category']]
        if not ((category == 'negative' and fraction == 0) or
                (category == 'positive' and fraction >= .02) or
                (category == 'hard_positive' and 0 < fraction < .02)):
            raise ValueError(f"Mask/category inconsistency: {row['sample_id']}")
        if self.augment:
            rgb, target = synchronized_transform(rgb, target, random.random() < .5,
                                                 random.random() < .5, random.randrange(4))
        image_tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().div(255)
        return {
            'image': (image_tensor - self.mean) / self.std,
            'mask': torch.from_numpy(target.copy()).unsqueeze(0),
            'sample_id': row['sample_id'], 'category': category,
            'uncertain_source': row.get('source_label_uncertain_filename', '').lower() == 'true',
        }


def pilot_split(rows, seed=42):
    """Choose groups before sampling; cap combined train+val at 20 per category."""
    if any(not row.get('source_scene') for row in rows):
        raise ValueError('Missing source_scene; establish an explicit split policy before training')
    rng = random.Random(seed)
    scenes = sorted({row['source_scene'] for row in rows})
    if len(scenes) < 2:
        raise ValueError('Cannot create a disjoint scene split from fewer than two scenes')
    rng.shuffle(scenes)
    validation_scenes = set(scenes[:max(1, round(len(scenes) * .2))])
    train, validation = [], []
    for category in CATEGORIES:
        pools = [sorted((r for r in rows if r['category'] == category and
                         ((r['source_scene'] in validation_scenes) == is_validation)),
                        key=lambda r: r['sample_id']) for is_validation in [False, True]]
        n_val = min(4, len(pools[1]))
        n_train = min(20 - n_val, len(pools[0]))
        n_val = min(20 - n_train, len(pools[1]))
        train.extend(rng.sample(pools[0], n_train))
        validation.extend(rng.sample(pools[1], n_val))
    if not train or not validation:
        raise ValueError('Empty training or validation split')
    assert not ({r['source_scene'] for r in train} & {r['source_scene'] for r in validation})
    return train, validation
