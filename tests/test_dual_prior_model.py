import unittest

import torch

from pvseg.dual_prior import DualPriorSegmenter, segmentation_loss


class DualPriorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.support_dino = torch.randn(1, 1024, 4, 4)
        self.support_geo = torch.randn(1, 512, 4, 4)
        self.query_dino = torch.randn(1, 1024, 4, 4)
        self.query_geo = torch.randn(1, 512, 4, 4)
        self.support_mask = torch.zeros(1, 1, 64, 64)
        self.support_mask[..., :32] = 1
        self.query_mask = torch.zeros(1, 1, 64, 64)
        self.query_mask[..., :32] = 1
        self.haar = torch.randn(1, 6, 4, 4)

    def run_model(self, model, haar=None):
        return model(self.support_dino, self.support_geo, self.support_mask,
                     self.query_dino, self.query_geo, (64, 64), haar)

    def test_feature_fusion_decoder_shapes_and_gradients(self):
        model = DualPriorSegmenter(channels=8)
        result = self.run_model(model)
        self.assertEqual(result['low_logits'].shape, (1, 1, 4, 4))
        self.assertEqual(result['logits'].shape, self.query_mask.shape)
        self.assertEqual(result['alpha'].shape, (1, 1, 4, 4))
        segmentation_loss(result['logits'], self.query_mask).backward()
        for name in ('dino_projection', 'geo_projection', 'feature_gate', 'decoder'):
            gradients = [parameter.grad for parameter in getattr(model, name).parameters()]
            self.assertTrue(any(gradient is not None and gradient.abs().sum() > 0
                                for gradient in gradients), name)
        self.assertIsNone(self.support_dino.grad)

    def test_frequency_modes_and_branch_controls(self):
        for frequency in ('none', 'score', 'feature'):
            for branch in ('dino', 'geo', 'both'):
                with self.subTest(frequency=frequency, branch=branch):
                    model = DualPriorSegmenter(channels=8, branch=branch,
                                               frequency=frequency)
                    result = self.run_model(model, None if frequency == 'none' else self.haar)
                    self.assertTrue(torch.isfinite(result['logits']).all())
                    if branch == 'dino':
                        self.assertTrue((result['alpha'] == 0).all())
                    elif branch == 'geo':
                        self.assertTrue((result['alpha'] == 1).all())
                    if frequency != 'none':
                        segmentation_loss(result['logits'], self.query_mask).backward()
                        module = (model.feature_modulation if frequency == 'feature'
                                  else model.score_correction)
                        self.assertTrue(any(parameter.grad is not None and
                                            parameter.grad.abs().sum() > 0
                                            for parameter in module.parameters()))

    def test_zero_initialized_haar_is_identity(self):
        model = DualPriorSegmenter(channels=8, frequency='feature')
        without_haar = DualPriorSegmenter(channels=8)
        without_haar.load_state_dict(model.state_dict())
        torch.testing.assert_close(self.run_model(model, self.haar)['logits'],
                                   self.run_model(without_haar)['logits'])


if __name__ == '__main__':
    unittest.main()
