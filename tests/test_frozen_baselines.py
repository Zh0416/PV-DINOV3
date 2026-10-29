import unittest

import numpy as np
import torch

from pvseg.frozen_baselines import (calibrate_threshold, confusion, foreground_prototype,
                                    foreground_background_prototypes, foreground_background_response,
                                    metrics, overlap_response, prototype_response,
                                    ranking_metrics, score_histogram)


class FrozenBaselineChecks(unittest.TestCase):
    def test_background_prototype_and_signed_margin(self):
        features = torch.tensor([[[[1., -1.], [1., -1.]], [[0., 0.], [0., 0.]]]])
        mask = torch.tensor([[[[1., 0.], [1., 0.]]]])
        foreground, background = foreground_background_prototypes(features, mask)
        torch.testing.assert_close(foreground, torch.tensor([[1., 0.]]))
        torch.testing.assert_close(background, torch.tensor([[-1., 0.]]))
        response = foreground_background_response(features, foreground, background, (2, 2))
        torch.testing.assert_close(response, torch.tensor([[[[1., -1.], [1., -1.]]]]))
        torch.testing.assert_close(foreground_background_response(features, foreground, foreground, (2, 2)),
                                   torch.zeros_like(response))
        with self.assertRaisesRegex(ValueError, 'no background'):
            foreground_background_prototypes(features, torch.ones_like(mask))
        with self.assertRaisesRegex(ValueError, 'finite weights'):
            foreground_background_prototypes(features, mask + 1)

    def test_area_weights_and_background_distractor(self):
        features = torch.tensor([[[[1., 1.]], [[0., 1.]]]])
        mask = torch.zeros(1, 1, 2, 4)
        mask[..., :, :2] = 1
        # A query matching support background still has a strong foreground
        # cosine, but its relative foreground/background margin is negative.
        foreground, background = foreground_background_prototypes(features, mask)
        query = torch.tensor([[[[1.]], [[1.]]]])
        self.assertGreater(prototype_response(query, foreground, (1, 1)).item(), .7)
        self.assertLess(foreground_background_response(query, foreground, background, (1, 1)).item(), 0)
        # A single labeled pixel survives area downsampling; all other pixels
        # contribute only to background, including the mixed patch remainder.
        mask.zero_()
        mask[..., 0, 0] = 1
        foreground, background = foreground_background_prototypes(features, mask)
        torch.testing.assert_close(foreground, torch.tensor([[1., 0.]]))
        torch.testing.assert_close(background, torch.nn.functional.normalize(torch.tensor([[7., 4.]]), dim=1))

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
