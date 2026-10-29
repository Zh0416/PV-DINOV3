import unittest

import numpy as np
import torch

from pvseg.frozen_baselines import (calibrate_threshold, confusion, foreground_prototype,
                                    metrics, overlap_response, prototype_response,
                                    ranking_metrics, score_histogram)


class FrozenBaselineChecks(unittest.TestCase):
    def test_small_support_and_query_response(self):
        features = torch.tensor([[[[1., 0.], [0., 0.]], [[0., 1.], [1., 1.]]]])
        mask = torch.zeros(1, 1, 8, 8)
        mask[..., 0, 0] = 1
        prototype = foreground_prototype(features, mask)
        torch.testing.assert_close(prototype, torch.tensor([[1., 0.]]))
        result = prototype_response(features, prototype, (2, 2))
        torch.testing.assert_close(result, torch.tensor([[[[1., 0.], [0., 0.]]]]))
        with self.assertRaises(ValueError):
            foreground_prototype(features, mask * 0)

    def test_overlap_grid_and_uncovered_pixels(self):
        result = overlap_response([.2, .8], [(0, 0, 2, 2), (1, 0, 3, 2)], (2, 3))
        np.testing.assert_allclose(result[0], [.2, .5, .8])
        with self.assertRaises(ValueError):
            overlap_response([.2], [(0, 0, 1, 1)], (2, 3))

    def test_calibration_and_negative_metrics(self):
        scores = np.array([-.4, -.2, .5, .7])
        labels = np.array([0, 0, 1, 1], bool)
        hist = score_histogram(scores, labels)
        calibrated = calibrate_threshold(hist)
        self.assertEqual(calibrated['calibration_micro_iou'], 1)
        self.assertEqual(metrics(confusion(scores, labels, calibrated['threshold']))['iou'], 1)
        self.assertEqual(ranking_metrics(hist)['pixel_ap_histogram'], 1)
        negative = metrics(confusion(scores[:2], labels[:2], calibrated['threshold']))
        self.assertIsNone(negative['iou'])
        self.assertEqual(negative['false_positive_rate'], 0)


if __name__ == '__main__':
    unittest.main()
