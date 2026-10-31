import unittest

import torch

from pvseg.greenhouse_prior import GreenhouseContrastPrior


def concept_bank():
    return {'pv': torch.tensor([[1., 0., 0.]]),
            'greenhouse': torch.tensor([[0., 1., 0.]]),
            'other': torch.tensor([[0., 0., 1.]])}


class GreenhousePriorChecks(unittest.TestCase):
    def test_confused_visual_branch_keeps_pv_and_suppresses_only_gh_evidence(self):
        # DINO gives all pixels the same positive response: only the semantic
        # evidence distinguishes PV, greenhouse, other and ambiguous regions.
        geo = torch.tensor([[[[1., 0., 0., .5]],
                             [[0., 1., 0., .5]],
                             [[0., 0., 1., 0.]]]])
        spatial = torch.full((1, 1, 1, 4), .4)
        out = GreenhouseContrastPrior(concept_bank())(geo, spatial)
        torch.testing.assert_close(out['semantic_penalty'],
                                   torch.tensor([[[[0., .2, 0., 0.]]]]))
        delta = out['baseline_score'] - out['fused_score']
        torch.testing.assert_close(delta, torch.tensor([[[[0., .14, 0., 0.]]]]))
        torch.testing.assert_close(out['spatial'], spatial)

    def test_no_forced_choice_on_empty_or_unrelated_regions(self):
        geo = torch.tensor([[[[0., 0.]], [[0., .4]], [[0., .8]]]])
        out = GreenhouseContrastPrior(concept_bank())(geo, torch.zeros(1, 1, 1, 2))
        self.assertEqual(out['semantic_penalty'].count_nonzero().item(), 0)
        torch.testing.assert_close(out['fused_score'], out['baseline_score'])

    def test_disabled_correction_reproduces_a3_and_score_scale(self):
        torch.manual_seed(7)
        geo = torch.nn.functional.normalize(torch.randn(2, 3, 3, 4), dim=1) * .7
        spatial = torch.rand(2, 1, 3, 4) * 2 - 1
        for alpha in (0., .7, 1.):
            out = GreenhouseContrastPrior(concept_bank(), semantic_weight=alpha,
                                          strength=0)(geo, spatial)
            expected = (1 - alpha) * spatial + alpha * geo[:, :1]
            torch.testing.assert_close(out['fused_score'], expected)

    def test_template_and_crop_averaging_keep_mean_cosine(self):
        bank = concept_bank()
        bank['pv'] = torch.tensor([[2., 0., 0.], [0., 2., 0.]])
        # Overlap mean of two orthogonal unit crop embeddings.
        geo = torch.tensor([[[[.5]], [[.5]], [[0.]]]])
        out = GreenhouseContrastPrior(bank, strength=0)(geo, torch.zeros(1, 1, 1, 1))
        self.assertAlmostEqual(out['pv_semantic'].item(), .5)

    def test_maps_are_independent_of_batch_and_can_upsample_to_896(self):
        model = GreenhouseContrastPrior(concept_bank())
        geo = torch.zeros(2, 3, 56, 56)
        geo[0, 0], geo[1, 1] = 1., 1.
        spatial = torch.zeros(2, 1, 56, 56)
        batch = model(geo, spatial)
        for i in range(2):
            single = model(geo[i:i + 1], spatial[i:i + 1])
            torch.testing.assert_close(batch['fused_score'][i:i + 1], single['fused_score'])
        up = model(geo[:1], spatial[:1], output_size=(896, 896))
        self.assertEqual(up['fused_cam'].shape, (1, 1, 896, 896))
        self.assertTrue(torch.isfinite(up['fused_cam']).all())

    def test_state_preserves_settings_and_rejects_wrong_concept_order(self):
        original = GreenhouseContrastPrior(concept_bank(), strength=.3, max_penalty=.1)
        restored = GreenhouseContrastPrior(concept_bank())
        restored.load_state_dict(original.state_dict())
        self.assertEqual(restored.strength, .3)
        self.assertEqual(restored.max_penalty, .1)
        wrong_order = GreenhouseContrastPrior(dict(reversed(list(concept_bank().items()))))
        with self.assertRaisesRegex(ValueError, 'order'):
            wrong_order.load_state_dict(original.state_dict())

    def test_invalid_embeddings_and_feature_space_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'at least one'):
            GreenhouseContrastPrior({k: v for k, v in concept_bank().items() if k != 'other'})
        bank = concept_bank()
        bank['greenhouse'] = torch.zeros(1, 3)
        with self.assertRaisesRegex(ValueError, 'nonzero'):
            GreenhouseContrastPrior(bank)
        model = GreenhouseContrastPrior(concept_bank())
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            model(torch.zeros(1, 1024, 2, 2), torch.zeros(1, 1, 2, 2))
        with self.assertRaisesRegex(ValueError, 'unit crop'):
            model(torch.ones(1, 3, 2, 2), torch.zeros(1, 1, 2, 2))
        with self.assertRaisesRegex(ValueError, 'finite'):
            model(torch.full((1, 3, 2, 2), float('nan')), torch.zeros(1, 1, 2, 2))


if __name__ == '__main__':
    unittest.main()
