import unittest

import torch

from pvseg.support_only_segmenter import _support_threshold, fit_from_support, predict


class SupportOnlySegmenterTests(unittest.TestCase):
    def test_support_threshold_uses_original_histogram_order(self):
        probability = torch.tensor([[[[.8, .2]]]])
        target = torch.tensor([[[[1., 0.]]]])
        threshold, iou, bins = _support_threshold(probability, target, bins=4)
        self.assertEqual((threshold, iou, bins), (.75, 1., 4))
        self.assertEqual(float(((probability >= threshold) == target.bool()).float().mean()), 1.)

    def test_support_mask_fits_fresh_heads_and_changes_query_prediction(self):
        features = torch.zeros(1, 2, 4, 4)
        features[:, 0, :, :2] = 1
        features[:, 1, :, 2:] = 1
        mask = torch.zeros(1, 1, 64, 64)
        mask[..., :32] = 1
        first = fit_from_support(features, mask, steps=30)
        second = fit_from_support(features, 1 - mask, steps=30)
        first_score, first_prediction = predict(first, features, (64, 64))
        second_score, second_prediction = predict(second, features, (64, 64))
        self.assertEqual(len(first['models']), 4)
        self.assertGreater(first_score[:, :16].mean(), first_score[:, 48:].mean())
        self.assertLess(second_score[:, :16].mean(), second_score[:, 48:].mean())
        self.assertTrue(first_prediction[:, :16].any())
        self.assertFalse(first_prediction[:, 48:].any())
        self.assertTrue(second_prediction[:, 48:].any())
        self.assertFalse(second_prediction[:, :16].any())
        self.assertNotEqual(float(first['models'][0]['network.weight'].abs().sum()), 0)
        self.assertIn('threshold', first)
        self.assertNotIn('source_checkpoint', first)


if __name__ == '__main__':
    unittest.main()
