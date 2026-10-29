"""Allocation checks use synthetic geometry; no private dataset is required."""
import unittest

from pvseg.data import CATEGORIES
from pvseg.sspnet.learned_protocol import episode_schedule, separate, training_plan


class LearnedProtocolChecks(unittest.TestCase):
    def test_reserved_support_and_protected_groups_never_train(self):
        pos, neg = list(CATEGORIES)[:2]

        def row(sid, x, category=pos, scene='train'):
            return dict(sample_id=sid, source_scene=scene, category=category,
                        width='896', height='896', image_sha256=sid,
                        transform=f'[0.00001,0,{x},0,-0.00001,1]')

        rows = [row('reserved', 0), row('touching', .001), row('p1', 1), row('p2', 2),
                row('n1', 3, neg), row('cal', 10, scene='cal'), row('query', 20, scene='query')]
        config = dict(support_group='train', records=[dict(sample_id='query')])
        audit = dict(groups={x: [x] for x in ('train', 'cal', 'query')}, roles={
            'support': [dict(sample_id='reserved', group='train')],
            'calibration': [dict(sample_id='cal', group='cal')],
            'query': [dict(sample_id='query', group='query')]})
        train, plan = training_plan(rows, config, audit)
        self.assertEqual(plan['training_ids'], ['n1', 'p1', 'p2'])
        schedule = episode_schedule(plan, 12)
        self.assertEqual(schedule, episode_schedule(plan, 12))
        by_id = {r['sample_id']: r for r in rows}
        for i, (q, s) in enumerate(schedule):
            self.assertTrue(separate(by_id[q], by_id[s]))
            self.assertIn(q, plan['positive_query_ids'] if i % 2 == 0 else plan['negative_query_ids'])
        # A malformed group audit must not defeat the per-image collision check.
        rows[-1]['transform'] = rows[2]['transform']
        with self.assertRaisesRegex(ValueError, 'collides'):
            training_plan(rows, config, audit)


if __name__ == '__main__':
    unittest.main()
