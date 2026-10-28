"""Separate PSP/episodic formats; omit only the externally verified SAT encoder."""

import random
import os
from pathlib import Path

import numpy as np
import torch


def rng_state():
    numpy_rng = np.random.get_state()
    return {'python': random.getstate(), 'numpy_kind': numpy_rng[0],
            'numpy_keys': torch.from_numpy(numpy_rng[1].astype('int64')),
            'numpy_pos': numpy_rng[2], 'numpy_gauss': numpy_rng[3:],
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(rng):
    random.setstate(rng['python'])
    np.random.set_state((rng['numpy_kind'], rng['numpy_keys'].numpy().astype('uint32'),
                         rng['numpy_pos'], *rng['numpy_gauss']))
    torch.set_rng_state(rng['torch'])
    if torch.cuda.is_available() and rng['cuda']:
        torch.cuda.set_rng_state_all(rng['cuda'])


def save(path, model, optimizer, precision, config, epoch, step, sampler_generator=None, history=None):
    episodic = hasattr(model, 'dmsf')
    weights = {k: v.cpu() for k, v in model.state_dict().items() if not (episodic and k.startswith('dino.'))}
    metadata = model.metadata if episodic else model.provenance
    state = {'format': 'sspnet_episode_v1' if episodic else 'sspnet_psp_v1',
             'state_dict': weights, 'metadata': metadata, 'config': config,
             'optimizer': optimizer.state_dict(), 'scaler': precision.scaler.state_dict(),
             'epoch': epoch, 'step': step, 'rng': rng_state(),
             'sampler_rng': sampler_generator.get_state() if sampler_generator is not None else None,
             'history': history}
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load(path, model, optimizer=None, precision=None, restore_random=False, sampler_generator=None,
         expected_config=None, evaluate=False):
    state = torch.load(path, map_location='cpu', weights_only=True)
    episodic = hasattr(model, 'dmsf')
    expected = 'sspnet_episode_v1' if episodic else 'sspnet_psp_v1'
    if state['format'] != expected:
        raise ValueError('Checkpoint stage mismatch')
    if expected_config is not None:
        from .training import validate_resume_config
        validate_resume_config(state['config'], expected_config, evaluate)
    if episodic and state['metadata'] != model.metadata:
        raise ValueError('Model/checkpoint dependencies or architecture mismatch')
    missing, unexpected = model.load_state_dict(state['state_dict'], strict=False)
    expected_missing = {k for k in model.state_dict() if episodic and k.startswith('dino.')}
    if set(missing) != expected_missing or unexpected:
        raise ValueError(f'Incomplete checkpoint: {missing}, {unexpected}')
    if not episodic:
        model.provenance = state['metadata']
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
    if precision is not None:
        precision.scaler.load_state_dict(state['scaler'])
    if restore_random:
        restore_rng(state['rng'])
        if sampler_generator is not None and state.get('sampler_rng') is not None:
            sampler_generator.set_state(state['sampler_rng'])
    return state
