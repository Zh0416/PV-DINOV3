"""Behavioral checks of trainable priors, masked attention and episodic isolation."""

import inspect
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from pvseg.runtime import Precision
from pvseg.sspnet import SSPNet, checkpoint
from pvseg.sspnet.data import eligible_support
from pvseg.sspnet.encoders import CNN_MEAN, CNN_STD, SAT_MEAN, SAT_STD, normalize_rgb
from pvseg.sspnet.losses import objective
from pvseg.sspnet.modules import DCMA, DMSF, ShotWeighting, hierarchical_prior, minmax, resize


class TinyCNN(nn.Module):
    mid_dim = 12
    provenance = {'architecture': 'synthetic'}

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 12, 1)
        self.bn = nn.BatchNorm2d(12)

    def features(self, x):
        f = self.bn(self.conv(resize(x, (8, 8))))
        return {'low': f[:, :4], 'mid': f, 'levels': (f, f[:, :6]), 'base_logits': f[:, :2]}


class TinyDINO(nn.Module):
    dim = 8
    metadata = {'architecture': 'synthetic'}

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 1)

    def forward(self, x):
        return self.conv(resize(x, (4, 4)))


def model(shots=1):
    return SSPNet(TinyCNN(), TinyDINO(), torch.randn(1, 10), {'prompt': 'synthetic'},
                  shots=shots, hidden=16, heads=4, query_chunk=13)


class SSPNetChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(9)

    def test_rgb_has_separate_reversible_normalizations(self):
        rgb = torch.rand(2, 3, 9, 7)
        outputs = []
        for mean, std in [(CNN_MEAN, CNN_STD), (SAT_MEAN, SAT_STD)]:
            normalized = normalize_rgb(rgb, mean, std)
            recovered = normalized * rgb.new_tensor(std)[None, :, None, None] + rgb.new_tensor(mean)[None, :, None, None]
            torch.testing.assert_close(recovered, rgb)
            outputs.append(normalized)
        self.assertFalse(torch.equal(*outputs))

    def test_dmsf_bounds_and_both_learnable_projection_paths(self):
        dmsf = DMSF(8, 10, 16)
        q, s = torch.randn(2, 8, 4, 4), torch.randn(2, 8, 4, 4)
        mask = torch.zeros(2, 1, 4, 4)
        mask[:, :, :2] = 1
        result = dmsf(q, [s], [mask], torch.randn(1, 10))
        self.assertAlmostEqual(float(result['gamma'].detach()), .66818777, places=6)
        for key in ['spatial', 'semantic', 'fused']:
            self.assertTrue(((result[key] >= 0) & (result[key] <= 1)).all())
        (result['fused'] * torch.rand_like(result['fused'])).sum().backward()
        for parameter in [dmsf.visual.net[0].weight, dmsf.text_to_dino[0].weight,
                          dmsf.text_hidden.net[0].weight, dmsf.beta]:
            self.assertGreater(float(parameter.grad.abs().sum()), 0)
        self.assertTrue(torch.equal(minmax(torch.ones(1, 1, 3, 3)), torch.zeros(1, 1, 3, 3)))

    def test_attention_background_invariance_empty_fallback_and_chunking(self):
        dcma = DCMA(16, 4, 3).eval()
        q, s = torch.randn(2, 7, 16), torch.randn(2, 11, 16)
        mask = torch.zeros(2, 11, dtype=torch.bool)
        mask[:, :3] = True
        first = dcma.attend(q, s, mask)
        changed = s.clone()
        changed[:, 3:] += 5
        torch.testing.assert_close(first, dcma.attend(q, changed, mask))
        dcma.query_chunk = 0
        torch.testing.assert_close(first, dcma.attend(q, s, mask), atol=1e-6, rtol=1e-5)
        self.assertTrue(torch.isfinite(dcma.attend(q, s, torch.zeros_like(mask))).all())

    def test_shot_weighting_is_permutation_equivariant_and_normalized(self):
        reweight = ShotWeighting(5)
        q, supports = torch.randn(2, 4, 4, 4), [torch.randn(2, 4, 4, 4) for _ in range(5)]
        weights, discrepancy = reweight(q, supports)
        order = [2, 0, 4, 1, 3]
        permuted, same_discrepancy = reweight(q, [supports[i] for i in order])
        torch.testing.assert_close(permuted, weights[:, order])
        torch.testing.assert_close(same_discrepancy, discrepancy)
        torch.testing.assert_close(weights.sum(1), torch.ones(2))
        self.assertTrue(torch.isfinite(weights).all())

    def test_hierarchy_reports_normalized_max_near_ties(self):
        q = torch.randn(2, 8, 4, 4)
        s = torch.randn(2, 8, 4, 4)
        mask = torch.zeros(2, 1, 4, 4)
        mask[:, :, :2] = 1
        fg, dominant, diagnostics = hierarchical_prior((q, q), [(s, s)], [mask], (4, 4))
        self.assertTrue(diagnostics['near_tie'].all())
        self.assertFalse(diagnostics['base_selected'].any())
        self.assertTrue((dominant >= 0).all())

    def test_empty_query_full_backward_and_frozen_encoders(self):
        net = model().train()
        self.assertFalse(net.cnn.training)
        self.assertFalse(net.dino.training)
        self.assertEqual(list(inspect.signature(net.forward).parameters), ['query', 'support_images', 'support_masks'])
        image = torch.rand(1, 3, 64, 64)
        mask = torch.zeros(1, 1, 1, 64, 64)
        mask[:, :, :, :32] = 1
        output = net(image, image[:, None], mask)
        loss, terms = objective(output, torch.zeros(1, 1, 64, 64))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is None for p in net.cnn.parameters()))
        self.assertTrue(all(p.grad is None for p in net.dino.parameters()))
        self.assertGreater(float(net.dmsf.visual.net[0].weight.grad.abs().sum()), 0)
        self.assertGreater(float(net.dcma.pseudo[-1].weight.grad.abs().sum()), 0)
        self.assertAlmostEqual(float(loss.detach()), float((terms['main_ce'] + terms['meta_dice'] +
            .2 * terms['pseudo_bce'] + .04 * terms['spatial_dice']).detach()), places=6)

    def test_pseudo_gradient_reaches_semantic_projection_after_zero_init_step(self):
        net = model().train()
        image = torch.rand(1, 3, 64, 64)
        masks = torch.ones(1, 1, 1, 64, 64)
        optimizer = torch.optim.SGD([p for p in net.parameters() if p.requires_grad], lr=.01)
        for _ in range(2):
            optimizer.zero_grad()
            out = net(image, image[:, None], masks)
            torch.nn.functional.binary_cross_entropy_with_logits(out['pseudo_logits'],
                torch.rand_like(out['pseudo_logits'])).backward()
            optimizer.step()
        self.assertGreater(float(net.dmsf.text_to_dino[0].weight.grad.abs().sum()), 0)

    def test_five_shot_forward_and_empty_support_rejection(self):
        net = model(5).eval()
        image = torch.rand(1, 3, 64, 64)
        supports = torch.rand(1, 5, 3, 64, 64)
        masks = torch.ones(1, 5, 1, 64, 64)
        with torch.no_grad():
            output = net(image, supports, masks)
        self.assertEqual(tuple(output['logits'].shape), (1, 2, 64, 64))
        with self.assertRaisesRegex(ValueError, 'disappeared'):
            net(image, supports, torch.zeros_like(masks))

    def test_episode_support_rejects_same_scene_id_overlap_or_hash(self):
        def row(sid, scene, x):
            return {'sample_id': sid, 'source_scene': scene, 'category': next(iter(__import__('pvseg.data', fromlist=['CATEGORIES']).CATEGORIES)),
                    'width': '896', 'height': '896', 'transform': f'[0.00001,0,{x},0,-0.00001,1]',
                    'image_sha256': sid}
        q, s = row('q', 'a', 0), row('s', 'b', 1)
        groups = {'a': 'a', 'b': 'b'}
        self.assertTrue(eligible_support(q, s, groups))
        for changed in [dict(s, sample_id='q'), dict(s, source_scene='a'),
                        dict(s, transform=q['transform']), dict(s, image_sha256='q')]:
            self.assertFalse(eligible_support(q, changed, groups))

    def test_checkpoint_restores_cnn_and_trainable_parts_but_omits_dino(self):
        net = model()
        precision = Precision('cpu')
        optimizer = torch.optim.SGD([p for p in net.parameters() if p.requires_grad], lr=.01)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.pt'
            checkpoint.save(path, net, optimizer, precision, {}, 0, 1)
            before = net.meta_head[-1].weight.detach().clone()
            with torch.no_grad():
                net.meta_head[-1].weight.add_(7)
            state = checkpoint.load(path, net, optimizer, precision, restore_random=True)
            torch.testing.assert_close(net.meta_head[-1].weight, before)
            self.assertFalse(any(k.startswith('dino.') for k in state['state_dict']))
            self.assertTrue(any(k.startswith('cnn.') for k in state['state_dict']))
            net.metadata['shots'] = 5
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                checkpoint.load(path, net)


if __name__ == '__main__':
    unittest.main()
