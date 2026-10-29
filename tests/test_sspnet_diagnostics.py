"""Check collapse measurements against explicit pixel counts and saturation."""
import unittest
import torch
from pvseg.sspnet.diagnostics import binary_report, pooled_reports


class DiagnosticChecks(unittest.TestCase):
    def test_saturated_background_is_a_failure_on_positive_mask(self):
        logits = torch.tensor([[[[20., 20.]], [[-20., -20.]]]])
        report = binary_report(logits, torch.tensor([[[[1., 0.]]]]))
        self.assertEqual((report['tp'], report['fp'], report['fn'], report['tn']), (0, 0, 1, 1))
        self.assertEqual(report['iou'], 0.)
        self.assertEqual(report['saturated_fraction'], 1.)
        self.assertIsNone(report['precision'])
        self.assertLess(report['foreground_probability'], 1e-6)

    def test_pooled_iou_uses_pixels_and_handles_empty_truth(self):
        a = binary_report(torch.tensor([[[[0., 1.]], [[1., 0.]]]]), torch.ones(1, 1, 1, 2))
        b = binary_report(torch.tensor([[[[1., 1.]], [[0., 0.]]]]), torch.zeros(1, 1, 1, 2))
        self.assertIsNone(b['iou'])
        self.assertEqual(pooled_reports([a, b])['iou'], .5)
