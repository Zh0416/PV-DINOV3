"""Metadata-only spatial grouping before bounded sample selection."""

from collections import Counter, defaultdict
import json
import math
import random


def bounds(row):
    transform = json.loads(row['transform'])
    if len(transform) not in (6, 9) or not all(math.isfinite(float(x)) for x in transform):
        raise ValueError('Invalid geographic transform')
    a, b, c, d, e, f = map(float, transform[:6])
    width, height = int(row['width']), int(row['height'])
    if width != 896 or height != 896 or a * e - b * d == 0:
        raise ValueError('Invalid tile footprint')
    corners = [(a*x+b*y+c, d*x+e*y+f) for x, y in [(0, 0), (width, 0), (0, height), (width, height)]]
    return (min(x for x, _ in corners), min(y for _, y in corners),
            max(x for x, _ in corners), max(y for _, y in corners))


def overlap(first, second):
    return min(first[2], second[2]) - max(first[0], second[0]) > 1e-10 and min(first[3], second[3]) - max(first[1], second[1]) > 1e-10


def scene_groups(rows):
    if not rows or any(not r.get('source_scene') or not r.get('source_image') for r in rows):
        raise ValueError('Scene and original image provenance required')
    crs = {r.get('crs') for r in rows}
    if crs != {'EPSG:4326'}:
        raise ValueError(f'Explicit CRS adaptation required: {crs}')
    scenes = sorted({r['source_scene'] for r in rows})
    parent = {s: s for s in scenes}

    def find(scene):
        while parent[scene] != scene:
            parent[scene] = parent[parent[scene]]
            scene = parent[scene]
        return scene

    def union(first, second):
        a, b = find(first), find(second)
        parent[max(a, b)] = min(a, b)

    tiles, sources, hashes = defaultdict(list), {}, {}
    for row in rows:
        scene = row['source_scene']
        tiles[scene].append(bounds(row))
        source = row['source_image'].replace('\\', '/').split('/')[-1].casefold()
        if source in sources:
            union(scene, sources[source])
        sources[source] = scene
        digest = row.get('image_sha256')
        if digest:
            if digest in hashes:
                union(scene, hashes[digest])
            hashes[digest] = scene
    envelopes = {s: (min(b[0] for b in bs), min(b[1] for b in bs),
                     max(b[2] for b in bs), max(b[3] for b in bs)) for s, bs in tiles.items()}
    edges = []
    for index, first in enumerate(scenes):
        for second in scenes[index + 1:]:
            if overlap(envelopes[first], envelopes[second]) and any(overlap(a, b) for a in tiles[first] for b in tiles[second]):
                union(first, second)
                edges.append([first, second])
    groups = defaultdict(list)
    for scene in scenes:
        groups[find(scene)].append(scene)
    return dict(groups), edges


def spatial_pilot_split(rows, categories, seed=42):
    groups, edges = scene_groups(rows)
    if len(groups) < 2:
        raise ValueError('Fewer than two independent spatial groups; training stopped')
    rng = random.Random(seed)
    keys = sorted(groups)
    rng.shuffle(keys)
    validation_keys = set(keys[:max(1, round(len(keys)*.2))])
    assignment = {s: ('validation' if k in validation_keys else 'train') for k, scenes in groups.items() for s in scenes}
    train, validation = [], []
    for category in categories:
        pools = [sorted((r for r in rows if r['category'] == category and assignment[r['source_scene']] == split),
                        key=lambda r: r['sample_id']) for split in ['train', 'validation']]
        n_val = min(4, len(pools[1]))
        n_train = min(20-n_val, len(pools[0]))
        n_val = min(20-n_train, len(pools[1]))
        train.extend(rng.sample(pools[0], n_train))
        validation.extend(rng.sample(pools[1], n_val))
    if not train or not validation:
        raise ValueError('Empty split after spatial grouping')
    collisions = [[a['sample_id'], b['sample_id']] for a in train for b in validation if overlap(bounds(a), bounds(b))]
    if collisions or set(r['sample_id'] for r in train) & set(r['sample_id'] for r in validation):
        raise AssertionError('Split contains duplicate IDs or intersecting geographic footprints')
    report = {'seed': seed, 'spatial_groups': groups, 'scene_assignment_before_sampling': assignment,
              'cross_scene_overlap_edges': edges, 'selected_cross_split_overlaps': collisions,
              'crs': 'EPSG:4326', 'policy': 'Union entire scenes sharing source basename, manifest image hash or intersecting tile geographic bounds; split groups before sampling.',
              'available_counts': {split: dict(Counter(r['category'] for r in rows if assignment[r['source_scene']] == split)) for split in ['train', 'validation']},
              'limitation': 'Uses supplied affine/CRS/provenance, not independently surveyed geography; no extra geographic buffer; rotated bounds are conservative.'}
    return train, validation, report
