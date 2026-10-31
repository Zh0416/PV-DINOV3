import unittest
import torch
from torch.nn import functional as F
from pvseg.dynamic_brightness_fusion import CAMWeight, convex_fusion, gate_inputs


class DynamicWeightTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.d, self.s = torch.rand(1, 1, 5, 6), torch.rand(1, 1, 5, 6)
        self.x = gate_inputs(self.d, self.s, self.d, self.d, self.d * .1)

    def test_initialization_recovers_original(self):
        for mode in ('fixed', 'global', 'local'):
            a = CAMWeight(mode)(self.x)
            torch.testing.assert_close(a, torch.full_like(a, .7), rtol=0, atol=0)
            up = lambda t: F.interpolate(t, (19, 21), mode='bilinear', align_corners=False)
            torch.testing.assert_close(convex_fusion(self.d, self.s, a, (19, 21)),
                                       .3 * up(self.d) + .7 * up(self.s), rtol=0, atol=1e-7)

    def test_convex_range_and_interpolation_order(self):
        a = torch.rand_like(self.d)
        y = convex_fusion(self.d, self.s, a, (19, 21))
        up = lambda t: F.interpolate(t, (19, 21), mode='bilinear', align_corners=False)
        self.assertTrue((y >= torch.minimum(up(self.d), up(self.s)) - 1e-7).all())
        self.assertTrue((y <= torch.maximum(up(self.d), up(self.s)) + 1e-7).all())
        torch.testing.assert_close(y, (1 - up(a)) * up(self.d) + up(a) * up(self.s))
        self.assertGreater(float((y - up((1-a)*self.d+a*self.s)).abs().max()), .001)
        torch.testing.assert_close(convex_fusion(self.d, self.d, a), self.d)

    def test_gradients_only_reach_weight_model(self):
        d, s = self.d.requires_grad_(), self.s.requires_grad_()
        model = CAMWeight('local')
        x = gate_inputs(d, s, d, d, d * .1)
        convex_fusion(d, s, model(x), (19, 21)).square().mean().backward()
        self.assertIsNone(d.grad)
        self.assertIsNone(s.grad)
        self.assertGreater(float(model.net[-1].weight.grad.norm()), 0)

    def test_invalid_alpha(self):
        with self.assertRaises(ValueError):
            convex_fusion(self.d, self.s, torch.full_like(self.d, 1.01))


if __name__ == '__main__':
    unittest.main()
