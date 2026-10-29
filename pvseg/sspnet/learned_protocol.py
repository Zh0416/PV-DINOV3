"""Metadata-only allocation for corrected A3; query performance never selects IDs."""

import hashlib
import json
import random

from pvseg.data import CATEGORIES
from pvseg.spatial import bounds, overlap


def separate(a, b):
    return (a['sample_id'] != b['sample_id'] and
            a['image_sha256'] != b['image_sha256'] and
            not overlap(bounds(a), bounds(b)))


def training_plan(rows, config, audit):
    """Only the historical support spatial group is eligible for new training.

    All named query groups and the calibration group are excluded as complete
    groups. Reserve inference supports and their intersecting/duplicate tiles.
    Within the one training scene, episode pairs must be nonoverlapping. This
    is an explicit known-PV adaptation, not a base/novel FSS benchmark.
    """
    group_for = {s: k for k, scenes in audit['groups'].items() for s in scenes}
    by_id = {r['sample_id']: r for r in rows}
    reserved = [by_id[r['sample_id']] for r in audit['roles']['support']]
    forbidden = {r['group'] for role in ('calibration', 'query') for r in audit['roles'][role]}
    if config['support_group'] in forbidden:
        raise ValueError('Training group intersects query/calibration groups')
    named = {r['sample_id'] for r in config['records']}
    training = sorted([r for r in rows if group_for[r['source_scene']] == config['support_group']
                       and r['sample_id'] not in named and all(separate(r, s) for s in reserved)],
                      key=lambda r: r['sample_id'])
    # A positive touching every other candidate cannot form a valid episode.
    # Remove it before either training stage; never relax spatial separation.
    positive_pool = [r for r in training if CATEGORIES[r['category']] != 'negative']
    excluded = [r['sample_id'] for r in training if not any(separate(r, s) for s in positive_pool)]
    training = [r for r in training if r['sample_id'] not in excluded]
    positives = [r for r in training if CATEGORIES[r['category']] != 'negative']
    negatives = [r for r in training if CATEGORIES[r['category']] == 'negative']
    candidates = {r['sample_id']: [s['sample_id'] for s in positives if separate(r, s)] for r in training}
    if not positives or not negatives or any(not v for v in candidates.values()):
        raise ValueError('Need positive/negative training queries with nonoverlapping positive supports')
    # Check across roles even when a supplied grouping/audit is malformed.
    protected = [by_id[r['sample_id']] for role in ('calibration', 'query') for r in audit['roles'][role]]
    if any(not separate(t, p) for t in training for p in protected):
        raise ValueError('Training collides with calibration/query image or footprint')
    result = {'training_ids': [r['sample_id'] for r in training],
              'positive_query_ids': [r['sample_id'] for r in positives],
              'negative_query_ids': [r['sample_id'] for r in negatives],
              'support_candidates': candidates,
              'reserved_inference_support_ids': [r['sample_id'] for r in reserved],
              'excluded_no_nonoverlapping_support': excluded,
              'training_group': config['support_group'], 'forbidden_groups': sorted(forbidden),
              'scope': 'supervised known-PV development adaptation; additional training masks; no novel-class claim',
              'pair_policy': 'within training group, distinct IDs/hashes and no footprint intersection',
              'query_sampling': 'alternate positive/negative; uniform within class; fixed training seed'}
    result['split_sha256'] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
    return training, result


def episode_schedule(plan, steps, seed=42):
    if steps < 1:
        raise ValueError('Training steps must be positive')
    rng = random.Random(seed)
    schedule = []
    for step in range(steps):
        pool = plan['positive_query_ids'] if step % 2 == 0 else plan['negative_query_ids']
        query = rng.choice(pool)
        schedule.append((query, rng.choice(plan['support_candidates'][query])))
    return schedule
