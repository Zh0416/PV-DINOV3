"""Check leakage-sensitive API and numerical invariants, not model accuracy."""
import unittest

import numpy as np

from pvseg.brightness_gate import (apply_brightness_gate, brightness_gate,
    brightness_grid, fit_brightness_reference)


class BrightnessGateTests(unittest.TestCase):
    def test_normalization_is_affine_exposure_invariant_without_clipping(self):
        image = np.repeat(np.arange(16, 80, dtype=np.uint8).reshape(8, 8, 1), 3, axis=2)
        raw, z, _ = brightness_grid(image, (2, 2))
        raw2, z2, _ = brightness_grid(image * 2 + 5, (2, 2))
        np.testing.assert_allclose(raw2, raw * 2 + 5, atol=3e-5)
        np.testing.assert_allclose(z2, z, atol=2e-6)

    def test_zero_one_and_ordered_predictions(self):
        pv = np.linspace(-.9, .9, 9, dtype=np.float32).reshape(3, 3)
        penalty = np.full_like(pv, .2)
        zero, _ = apply_brightness_gate(pv, penalty, np.zeros_like(pv))
        one, _ = apply_brightness_gate(pv, penalty, np.ones_like(pv))
        half, _ = apply_brightness_gate(pv, penalty, np.full_like(pv, .5))
        np.testing.assert_array_equal(zero, pv)
        np.testing.assert_array_equal(one, np.clip(pv - penalty, -1, 1))
        self.assertTrue((one <= half).all() and (half <= zero).all())

    def test_equal_image_weights_and_negative_mask_ignored(self):
        z = [np.array([[0., 0.]]), np.array([[1., 1.]]), np.array([[100., 100.]])]
        areas = [np.array([[1., 1.]]), np.array([[.01, .01]]), np.zeros((1, 2))]
        ref = fit_brightness_reference(z, areas, (.25, .75))
        self.assertEqual(ref['lower'], 0.)
        self.assertEqual(ref['upper'], 1.)
        self.assertEqual(ref['positive_calibration_images'], 2)

    def test_monotone_gate_and_constant_image_fallback(self):
        _, z, meta = brightness_grid(np.full((8, 8, 3), 50, dtype=np.uint8), (2, 2))
        self.assertEqual(meta['scale_dn'], 1.)
        ref = fit_brightness_reference([z], [np.ones_like(z)])
        self.assertGreater(ref['upper'], ref['lower'])
        np.testing.assert_array_equal(brightness_gate(np.array([-1., 0., .0005, 1.]), ref),
                                      np.array([0., 0., .5, 1.]))

    def test_invalid_normalized_rgb_and_query_mask_free_interface(self):
        with self.assertRaises(ValueError):
            brightness_grid(np.zeros((8, 8, 3), np.float32), (2, 2))
        with self.assertRaises(ValueError):
            apply_brightness_gate(np.ones(3), np.ones(3), np.array([0., 1., 2.]))


if __name__ == '__main__':
    unittest.main()
