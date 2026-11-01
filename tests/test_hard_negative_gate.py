import unittest

import torch
import numpy as np
from pvseg.local_texture import texture_features

from pvseg.hard_negative_gate import ResidualFusionGate, residual_fusion


class HardNegativeGateTests(unittest.TestCase):
    def test_zero_initialization_matches_original_fusion(self):
        torch.manual_seed(4)
        d, s = torch.rand(2, 1, 8, 9), torch.rand(2, 1, 8, 9)
        x = torch.randn(2, 6, 8, 9)
        gate = ResidualFusionGate(6)
        alpha, residual = gate(x)
        torch.testing.assert_close(alpha, torch.full_like(alpha, .7), rtol=0, atol=0)
        torch.testing.assert_close(residual, torch.zeros_like(residual), rtol=0, atol=0)
        torch.testing.assert_close(residual_fusion(d, s, alpha, residual), .3 * d + .7 * s)

    def test_residual_is_bounded_and_gradients_reach_gate(self):
        gate = ResidualFusionGate(3, max_residual=.2)
        alpha, residual = gate(torch.randn(1, 3, 5, 6))
        self.assertTrue(((alpha >= 0) & (alpha <= 1)).all())
        self.assertLessEqual(float(residual.detach().abs().max()), .2)
        (alpha.mean() + residual.mean()).backward()
        self.assertGreater(float(gate.residual_head.bias.grad.detach().abs().sum()), 0)

    def test_residual_corrects_joint_false_positive_without_encoder_gradient(self):
        gate = ResidualFusionGate(3)
        d = torch.full((1, 1, 5, 6), .3, requires_grad=True)
        s = torch.full((1, 1, 5, 6), .3, requires_grad=True)
        alpha, residual = gate(torch.ones(1, 3, 5, 6))
        score = residual_fusion(d, s, alpha, residual, (20, 24))
        torch.nn.functional.softplus(20 * (score - .2)).mean().backward()
        self.assertIsNone(d.grad)
        self.assertIsNone(s.grad)
        self.assertEqual(float(gate.alpha_head.bias.grad), 0.)
        # Gradient descent can decrease the residual even when D == S.
        self.assertGreater(float(gate.residual_head.bias.grad), 0.)

    def test_texture_constant_image_is_finite_and_invalid_rgb_rejected(self):
        features = texture_features(np.zeros((896, 896, 3), dtype=np.uint8))
        self.assertEqual(features.shape, (9, 56, 56))
        self.assertTrue(np.isfinite(features).all())
        self.assertEqual(float(np.abs(features).max()), 0.)
        with self.assertRaises(ValueError):
            texture_features(np.zeros((896, 896, 3), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
