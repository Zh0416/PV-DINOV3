import unittest

import torch

from pvseg.losses import segmentation_loss


class SegmentationLossChecks(unittest.TestCase):
    def test_false_positive_and_false_negative_receive_opposite_gradients(self):
        logits = torch.tensor([[[[2.0, -2.0]]]], requires_grad=True)
        target = torch.tensor([[[[0.0, 1.0]]]])

        loss = segmentation_loss(logits, target)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(logits.grad[0, 0, 0, 0].item(), 0.0)
        self.assertLess(logits.grad[0, 0, 0, 1].item(), 0.0)

    def test_fractional_mask_is_area_resized(self):
        logits = torch.zeros(1, 1, 1, 1, requires_grad=True)
        mask = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])

        loss = segmentation_loss(logits, mask)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())


if __name__ == '__main__':
    unittest.main()
