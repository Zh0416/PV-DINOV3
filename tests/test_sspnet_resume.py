"""Interrupted-run equivalence and refusal of unsafe resume/output reuse."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from scripts import sspnet as cli
from pvseg.sspnet import checkpoint
from pvseg.sspnet.training import check_output_directory, resume_history


class SmallBase(nn.Module):
    def __init__(self, imagenet=False):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 8, 1), nn.ReLU(), nn.Dropout2d(.25), nn.Conv2d(8, 2, 1))
        self.provenance = {'architecture': 'resume_fixture'}

    def forward(self, image):
        return self.net(image)


class SmallData(Dataset):
    def __init__(self, root, rows, size, augment=False):
        self.augment = augment

    def __len__(self):
        return 3

    def __getitem__(self, index):
        image = torch.full((3, 16, 16), index / 3)
        if self.augment:
            image += torch.rand_like(image) * .1 + float(np.random.rand()) * .01
        mask = torch.zeros(1, 16, 16)
        mask[:, :4 + index, :8] = 1
        return {'image': image, 'mask': mask, 'category': 'positive'}


class ResumeChecks(unittest.TestCase):
    def test_epoch_interrupt_matches_uninterrupted_training_and_keeps_records(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'manifest.csv').write_text('fixture', encoding='utf-8')
            full, resumed = root / 'full', root / 'resumed'
            split = {'split_sha256': 'fixture'}
            common = ['base', '--data-root', str(root), '--device', 'cpu', '--epochs', '3',
                      '--max-steps', '3', '--accumulation', '2', '--size', '64']
            with patch.object(cli, 'read_manifest', return_value=[]), \
                 patch.object(cli, 'make_split', return_value=([{}, {}, {}], [{}], split)), \
                 patch.object(cli, 'PSPNet', SmallBase), patch.object(cli, 'ResizedPV', SmallData), \
                 patch.object(cli, 'environment', return_value={'fixture': True}), redirect_stdout(io.StringIO()):
                cli.main([*common, '--output', str(full)])
                full_rng = torch.get_rng_state().clone()
                original_save = checkpoint.save

                def interrupt(*args, **kwargs):
                    original_save(*args, **kwargs)
                    raise InterruptedError('simulated process stop after checkpoint, before history.json')

                with patch.object(checkpoint, 'save', side_effect=interrupt), self.assertRaises(InterruptedError):
                    cli.main([*common, '--output', str(resumed)])
                self.assertFalse((resumed / 'history.json').exists())
                metadata = {p.name: p.read_bytes() for p in resumed.glob('*.json')}
                incompatible = [*common, '--output', str(resumed), '--resume', str(resumed / 'last.pt'),
                                '--max-steps', '2']
                with self.assertRaisesRegex(ValueError, 'max_steps'):
                    cli.main(incompatible)
                self.assertEqual(metadata, {p.name: p.read_bytes() for p in resumed.glob('*.json')})
                cli.main([*common, '--output', str(resumed), '--resume', str(resumed / 'last.pt')])
                self.assertTrue(torch.equal(full_rng, torch.get_rng_state()))
            a = torch.load(full / 'last.pt', weights_only=True)
            b = torch.load(resumed / 'last.pt', weights_only=True)
            self.assertEqual((a['step'], b['step']), (6, 6))
            for key in a['state_dict']:
                self.assertTrue(torch.equal(a['state_dict'][key], b['state_dict'][key]), key)
            for key in a['optimizer']['state']:
                for name, value in a['optimizer']['state'][key].items():
                    torch.testing.assert_close(value, b['optimizer']['state'][key][name], atol=0, rtol=0)
            self.assertEqual(a['history'], b['history'])
            self.assertEqual(json.loads((full / 'history.json').read_text()),
                             json.loads((resumed / 'history.json').read_text()))

    def test_legacy_history_is_preserved_and_missing_history_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'last.pt'
            state = {'epoch': 1}
            with self.assertRaisesRegex(ValueError, 'history missing'):
                resume_history(state, path)
            history = [{'epoch': i, 'loss': 1 / (i + 1)} for i in range(3)]
            path.with_name('history.json').write_text(json.dumps(history))
            self.assertEqual(resume_history(state, path), history[:2])
            state['history'] = [history[1]]
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                resume_history(state, path)

    def test_existing_output_cannot_be_restarted_without_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'config.json').write_text('{}')
            with self.assertRaisesRegex(RuntimeError, 'Experiment output exists'):
                check_output_directory(directory)
            check_output_directory(directory, resume=True)


if __name__ == '__main__':
    unittest.main()
