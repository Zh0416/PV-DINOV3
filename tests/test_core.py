"""Synthetic regression checks; no real data or pretrained weights required."""

import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image
import torch

from pvseg.data import CATEGORIES, PV896Dataset, pilot_split, synchronized_transform
from pvseg.losses import binary_loss
from pvseg.metrics import BinaryCounts
from pvseg.runtime import load_checkpoint, save_checkpoint


class CoreChecks(unittest.TestCase):
    def test_all_discrete_augmentations_preserve_alignment(self):
        mask = np.zeros((16, 16), dtype=np.float32)
        mask[1:5, 7:12] = 1
        image = np.repeat((mask * 255).astype('uint8')[..., None], 3, axis=2)
        for horizontal in [False, True]:
            for vertical in [False, True]:
                for rotation in range(4):
                    rgb, target = synchronized_transform(image, mask, horizontal, vertical, rotation)
                    np.testing.assert_array_equal(rgb[..., 0] == 255, target == 1)

    def test_unknown_label_values_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.fromarray(np.zeros((16, 16, 3), dtype='uint8')).save(root / 'sample.bmp')
            row = {'sample_id': 'sample', 'category': next(iter(CATEGORIES)),
                   'image': 'sample.bmp', 'label': 'mask.bmp'}
            for code in [1, 128]:
                Image.fromarray(np.full((16, 16), code, dtype='uint8')).save(root / 'mask.bmp')
                dataset = PV896Dataset(root, [row], [0, 0, 0], [1, 1, 1], img_size=16)
                with self.assertRaisesRegex(ValueError, 'encoding'):
                    dataset[0]

    def test_empty_negative_does_not_get_perfect_foreground_iou(self):
        counts = BinaryCounts()
        counts.update(torch.full((1, 1, 4, 4), -20.), torch.zeros(1, 1, 4, 4))
        self.assertIsNone(counts.report()['iou'])
        counts.update(torch.full((1, 1, 4, 4), 20.), torch.zeros(1, 1, 4, 4))
        self.assertEqual(counts.report()['iou'], 0)
        self.assertEqual(counts.report()['negative_false_positive_images'], 1)
        self.assertEqual(counts.report()['negative_false_positive_pixels'], 16)
        self.assertEqual(counts.report()['negative_false_positive_pixel_ratio'], .5)

    def test_missed_positive_images_and_false_negative_pixels(self):
        counts = BinaryCounts()
        truth = torch.zeros(2, 1, 4, 4)
        truth[0, 0, 1:3, 1:3] = 1
        counts.update(torch.full_like(truth, -20.), truth)
        report = counts.report()
        self.assertEqual(report['fn'], 4)
        self.assertEqual(report['positive_images_without_true_positive_pixels'], 1)
        self.assertEqual(report['foreground_false_negative_ratio'], 1)
        self.assertEqual(report['empty_targets'], 1)
        self.assertEqual(report['negative_false_positive_pixels'], 0)

    def test_group_split_is_disjoint_and_capped(self):
        rows = [{'sample_id': f'{c}-{scene}-{i}', 'category': category, 'source_scene': f'scene{scene}'}
                for c, category in enumerate(CATEGORIES) for scene in range(8) for i in range(10)]
        train, val = pilot_split(rows)
        self.assertFalse({r['source_scene'] for r in train} & {r['source_scene'] for r in val})
        self.assertLessEqual(len(train) + len(val), 60)
        self.assertEqual((train, val), pilot_split(rows))

    def test_loss_is_finite_for_empty_and_nonempty_targets(self):
        for value in [0., 1.]:
            logits = torch.zeros(2, 1, 16, 16, requires_grad=True)
            loss = binary_loss(logits, torch.full_like(logits, value))
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(torch.isfinite(logits.grad).all())

    def test_safe_checkpoint_loader_restores_decoder(self):
        model = SimpleNamespace(decoder=torch.nn.Linear(2, 1), metadata={
            'backbone_id': 'fixture', 'weight_source': {'sha256': 'fixture'},
            'normalization': {}, 'decoder': {}, 'indices_zero_based': [], 'intermediate_norm': True})
        optimizer = torch.optim.AdamW(model.decoder.parameters())
        expected = model.decoder.weight.detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'decoder.pt'
            save_checkpoint(path, model, optimizer, {'seed': 42}, {'torch': str(torch.__version__)})
            with torch.no_grad():
                model.decoder.weight.add_(1)
            load_checkpoint(path, model, optimizer)
            torch.testing.assert_close(model.decoder.weight, expected)


if __name__ == '__main__':
    unittest.main()
