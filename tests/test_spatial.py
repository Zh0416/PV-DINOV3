"""Spatial leakage regression checks using only metadata."""

import json
import unittest

from pvseg.spatial import bounds, scene_groups, spatial_pilot_split


def tile(scene, x, category='positive', index=0):
    return {'sample_id': f'{scene}-{category}-{index}', 'source_scene': scene,
            'source_image': f'/original/{scene}.tif', 'crs': 'EPSG:4326',
            'width': '896', 'height': '896', 'category': category,
            'transform': json.dumps([.00001, 0, x, 0, -.00001, 22, 0, 0, 1])}


class SpatialChecks(unittest.TestCase):
    def test_cross_scene_overlap_chain_is_one_group(self):
        rows = [tile('a', 110), tile('b', 110.005), tile('c', 110.010), tile('d', 112)]
        groups, edges = scene_groups(rows)
        self.assertEqual(groups['a'], ['a', 'b', 'c'])
        self.assertEqual(len(edges), 2)

    def test_source_and_duplicate_image_provenance_unite_scenes(self):
        rows = [tile('a', 110), tile('b', 112), tile('c', 114)]
        rows[1]['source_image'] = rows[0]['source_image']
        rows[1]['image_sha256'] = rows[2]['image_sha256'] = 'same-image'
        self.assertEqual(list(scene_groups(rows)[0].values()), [['a', 'b', 'c']])

    def test_split_precedes_sampling_and_is_capped(self):
        rows = [tile(f's{s}', 100+s, c, i) for s in range(5)
                for c in ['positive', 'negative', 'hard_positive'] for i in range(25)]
        train, val, report = spatial_pilot_split(rows, ['positive', 'negative', 'hard_positive'])
        self.assertEqual((len(train), len(val)), (48, 12))
        self.assertEqual((train, val, report), spatial_pilot_split(rows, ['positive', 'negative', 'hard_positive']))
        self.assertFalse({r['source_scene'] for r in train} & {r['source_scene'] for r in val})
        self.assertEqual(report['selected_cross_split_overlaps'], [])

    def test_unusable_geography_stops_instead_of_fallback(self):
        rows = [tile('a', 110), tile('b', 112)]
        rows[1]['crs'] = 'unknown'
        with self.assertRaises(ValueError):
            scene_groups(rows)
        with self.assertRaises(ValueError):
            spatial_pilot_split([tile('a', 110)], ['positive'])


if __name__ == '__main__':
    unittest.main()
