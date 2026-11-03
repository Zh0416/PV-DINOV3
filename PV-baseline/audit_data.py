"""Audit PV-896 pixels and freeze spatially grouped PFENet baseline splits."""

import csv
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from pvseg.spatial import bounds, overlap, scene_groups


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    config = json.loads((HERE / 'config.json').read_text(encoding='utf-8'))
    root = (HERE / config['dataset_root']).resolve()
    manifest = root / config['manifest']
    with manifest.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows or len({row['sample_id'] for row in rows}) != len(rows):
        raise ValueError('Empty manifest or duplicate sample IDs')
    categories = Counter()
    empty_masks = 0
    image_digests = Counter()
    label_digests = Counter()
    nonbinary_masks = []
    corrupt = []
    metadata_mismatch = []
    category_mismatch = []
    for index, row in enumerate(rows, 1):
        sample_id = row['sample_id']
        image_path = (root / row['image']).resolve()
        label_path = (root / row['label']).resolve()
        if (not image_path.is_relative_to(root) or not label_path.is_relative_to(root)
                or image_path.stem != sample_id or label_path.stem != sample_id):
            raise ValueError(f'Unsafe or mismatched paths for {sample_id}')
        try:
            with Image.open(image_path) as image, Image.open(label_path) as label:
                image.load()
                label.load()
                if image.mode != 'RGB' or label.mode != 'L' or image.size != (896, 896) or label.size != (896, 896):
                    corrupt.append(sample_id)
                    continue
                mask = np.asarray(label)
                if not np.isin(mask, (0, 255)).all():
                    nonbinary_masks.append(sample_id)
                    continue
                pixels = int(np.count_nonzero(mask == 255))
                fraction = pixels / mask.size
                expected_category = ('负样本' if pixels == 0 else
                                     '正样本' if fraction >= .02 else '困难正样本')
                if row['category'] != expected_category:
                    category_mismatch.append(sample_id)
                if pixels != int(row['label_pixels']):
                    metadata_mismatch.append(sample_id)
                empty_masks += pixels == 0
                categories[row['category']] += 1
        except (OSError, ValueError):
            corrupt.append(sample_id)
            continue
        image_digest = sha256(image_path)
        label_digest = sha256(label_path)
        image_digests[image_digest] += 1
        label_digests[label_digest] += 1
        if image_digest != row['image_sha256'] or label_digest != row['label_sha256']:
            metadata_mismatch.append(sample_id)
        if index % 500 == 0:
            print(f'Audited {index}/{len(rows)} pairs', flush=True)
    groups, edges = scene_groups(rows)
    keys = sorted(groups)
    random.Random(config['split_seed']).shuffle(keys)
    counts = config['split_group_counts']
    if sum(counts.values()) != len(keys):
        raise ValueError('Split group counts do not cover every independent spatial group')
    test_keys = keys[:counts['test']]
    validation_keys = keys[counts['test']:counts['test'] + counts['validation']]
    train_keys = keys[counts['test'] + counts['validation']:]
    assignments = {'train': train_keys, 'validation': validation_keys, 'test': test_keys}
    group_for_scene = {scene: key for key, scenes in groups.items() for scene in scenes}
    split_for_group = {key: split for split, members in assignments.items() for key in members}
    split_ids = {split: sorted(row['sample_id'] for row in rows
                               if split_for_group[group_for_scene[row['source_scene']]] == split)
                 for split in assignments}
    rows_by_id = {row['sample_id']: row for row in rows}
    support_counts = {}
    for split, ids in split_ids.items():
        support_counts[split] = {}
        for key in assignments[split]:
            positives = sum(rows_by_id[sample_id]['category'] != '负样本' and
                            group_for_scene[rows_by_id[sample_id]['source_scene']] != key
                            for sample_id in ids)
            support_counts[split][key] = positives
            if positives < 5:
                raise ValueError(f'{split} group {key} has fewer than five eligible supports')
    for first, second in (('train', 'validation'), ('train', 'test'), ('validation', 'test')):
        first_rows = [rows_by_id[sample_id] for sample_id in split_ids[first]]
        second_rows = [rows_by_id[sample_id] for sample_id in split_ids[second]]
        if (set(split_ids[first]) & set(split_ids[second]) or
                {row['source_image'] for row in first_rows} & {row['source_image'] for row in second_rows} or
                {row['image_sha256'] for row in first_rows} & {row['image_sha256'] for row in second_rows} or
                any(overlap(bounds(left), bounds(right)) for left in first_rows for right in second_rows)):
            raise ValueError(f'Spatial leakage between {first} and {second}')
    split = dict(seed=config['split_seed'], source_manifest_sha256=sha256(manifest),
                 groups=groups, assignments=assignments, sample_ids=split_ids)
    audit = dict(status='PASS' if not (corrupt or nonbinary_masks or
                                      metadata_mismatch or category_mismatch) else 'FAIL',
                 manifest_rows=len(rows), checked_pairs=sum(categories.values()),
                 category_counts=dict(categories), empty_masks=empty_masks,
                 duplicate_image_hash_groups=sum(value > 1 for value in image_digests.values()),
                 duplicate_label_hash_groups=sum(value > 1 for value in label_digests.values()),
                 corrupt_or_wrong_shape=corrupt, nonbinary_masks=nonbinary_masks,
                 category_mismatch=category_mismatch,
                 metadata_or_checksum_mismatch=metadata_mismatch,
                 source_scenes=len({row['source_scene'] for row in rows}),
                 independent_spatial_groups=len(groups), cross_scene_overlap_edges=len(edges),
                 split_counts={name: len(ids) for name, ids in split_ids.items()},
                 eligible_support_counts_by_query_group=support_counts,
                 split_categories={name: dict(Counter(rows_by_id[sample_id]['category'] for sample_id in ids))
                                   for name, ids in split_ids.items()},
                 split_spatial_leakage='none under manifest source, hash and bounding-box checks')
    (HERE / 'reports').mkdir(exist_ok=True)
    (HERE / 'splits').mkdir(exist_ok=True)
    (HERE / 'reports/data_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    (HERE / 'splits/pv896_seed321.json').write_text(json.dumps(split, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(audit, ensure_ascii=False), flush=True)
    if audit['status'] != 'PASS':
        raise RuntimeError('PV-896 data audit failed')


if __name__ == '__main__':
    main()
