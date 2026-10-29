import unittest

import numpy as np
import torch

from pvseg.frozen_baselines import (calibrate_threshold, confusion, foreground_prototype,
                                    aligned_crop_features, calibrate_fusion, fuse_responses,
                                    foreground_background_prototypes, foreground_background_response,
                                    metrics, overlap_response, prototype_response,
                                    ranking_metrics, score_histogram)


class FrozenBaselineChecks(unittest.TestCase):
    def test_crop_feature_alignment_preserves_native_semantic_scores(self):
        vectors = torch.tensor([[1., 0.], [0., 1.]])
        boxes = [(0, 0, 4, 4), (2, 0, 6, 4)]
        grid = aligned_crop_features(vectors, boxes, (4, 6), (2, 3))
        torch.testing.assert_close(grid[0, :, 0], torch.tensor([[1., .5, 0.], [0., .5, 1.]]))
        text = torch.nn.functional.normalize(torch.tensor([1., 2.]), dim=0)
        actual = (grid * text[None, :, None, None]).sum(1)[0].numpy()
        native = overlap_response((vectors @ text).numpy(), boxes, (4, 6))
        np.testing.assert_allclose(actual, native.reshape(2, 2, 3, 2).mean((1, 3)))
        with self.assertRaisesRegex(ValueError, 'align'):
            aligned_crop_features(vectors, [(0, 0, 3, 4), (2, 0, 6, 4)], (4, 6), (2, 3))

    def test_fusion_feature_readout_endpoints_and_calibration(self):
        dino = np.array([1., 0.], np.float32)
        clip = np.array([.3, .4], np.float32)
        fg, bg, text = np.array([1., 0.]), np.array([0., 1.]), np.array([0., 1.])
        spatial, semantic = np.array([dino @ (fg - bg) / 2]), np.array([clip @ text])
        alpha = .3
        joined = np.r_[np.sqrt(1-alpha)*dino, np.sqrt(alpha)*clip]
        foreground = np.r_[np.sqrt(1-alpha)*fg, np.sqrt(alpha)*text]
        background = np.r_[np.sqrt(1-alpha)*bg, -np.sqrt(alpha)*text]
        np.testing.assert_allclose(fuse_responses(spatial, semantic, alpha), [joined @ (foreground-background) / 2])
        np.testing.assert_array_equal(fuse_responses(spatial, semantic, 0), spatial)
        np.testing.assert_array_equal(fuse_responses(spatial, semantic, 1), semantic)
        spatial = np.array([.8, .7, .6, .5], np.float32)
        semantic = np.array([-.8, -.7, .6, .5], np.float32)
        target = np.array([0, 0, 1, 1], bool)
        fitted = calibrate_fusion({42: [spatial], 43: [spatial]}, [semantic], [target], [0., .5, 1.])
        self.assertEqual(fitted['selected_semantic_weight'], .5)
        self.assertEqual(fitted['selected_runs']['42']['calibration_micro_iou'], 1)
        self.assertEqual(fitted['selected_runs']['42'], fitted['selected_runs']['43'])
        with self.assertRaises(ValueError):
            fuse_responses(spatial, semantic, -1)

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
