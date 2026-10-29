"""Regression checks for saturation correction and absent foreground."""
import unittest

import torch
from torch.nn import functional as F

from pvseg.sspnet.losses import balanced_meta_ce


class BalancedMetaLossChecks(unittest.TestCase):
    def test_saturated_false_negative_keeps_corrective_gradient(self):
        logits = torch.tensor([[[[150., 150.]], [[-150., -150.]]]], requires_grad=True)
        mask = torch.tensor([[[[1., 0.]]]])
        loss = balanced_meta_ce(logits, mask)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertLess(float(logits.grad[0, 1, 0, 0]), -.49)
        self.assertGreater(float(logits.grad[0, 0, 0, 0]), .49)

    def test_duplicate_background_does_not_reduce_foreground_weight(self):
        short = torch.tensor([[[[0., 2.]], [[1., 0.]]]])
        long = torch.cat([short[..., :1], short[..., 1:].expand(1, 2, 1, 17)], -1)
        mask = torch.zeros(1, 1, 1, 18)
        mask[..., 0] = 1
        torch.testing.assert_close(balanced_meta_ce(short, mask[..., :2]), balanced_meta_ce(long, mask))

    def test_negative_only_matches_background_ce_and_is_finite(self):
        logits = torch.randn(2, 2, 5, 7, requires_grad=True)
        mask = torch.zeros(2, 1, 5, 7)
        loss = balanced_meta_ce(logits, mask)
        torch.testing.assert_close(loss, F.cross_entropy(logits, mask[:, 0].long()))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())


if __name__ == '__main__':
    unittest.main()
