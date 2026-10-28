"""Resume preflight checks shared by the CLI and convergence diagnostics."""

import json
from pathlib import Path


TRAIN_RESUME_KEYS = ('stage', 'split_sha256', 'size', 'shots', 'lr', 'accumulation',
                     'batch_size', 'epochs', 'max_steps', 'seed', 'workers',
                     'query_chunk', 'routing', 'pilot', 'allow_untrained_base', 'precision')
EVAL_RESUME_KEYS = ('split_sha256', 'size', 'shots', 'routing')


def validate_resume_config(saved, current, evaluate=False):
    for key in EVAL_RESUME_KEYS if evaluate else TRAIN_RESUME_KEYS:
        if key not in saved or saved[key] != current[key]:
            raise ValueError(f'Resume config differs: {key}')


def resume_history(state, checkpoint_path):
    """The checkpoint is authoritative; recover legacy adjacent JSON if needed.

    Ignore JSON epochs newer than the checkpoint (a partially written run), and
    refuse missing earlier records rather than silently starting an empty log.
    """
    history = state.get('history')
    if history is None:
        path = Path(checkpoint_path).with_name('history.json')
        if not path.is_file():
            raise ValueError('Resume history missing: use a checkpoint with embedded history or adjacent history.json')
        history = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(history, list) or any(not isinstance(h, dict) or 'epoch' not in h for h in history):
        raise ValueError('Invalid resume history')
    history = [h for h in history if h['epoch'] <= state['epoch']]
    if [h['epoch'] for h in history] != list(range(state['epoch'] + 1)):
        raise ValueError('Resume history is incomplete or epochs are out of order')
    return history


def check_output_directory(output, resume=False):
    """Never reuse experiment metadata accidentally, even after a failed run."""
    output = Path(output)
    if (output / 'completion.json').exists():
        raise RuntimeError('Completed output exists; use a new output directory')
    if not resume and any((output / name).exists() for name in
                          ('config.json', 'split.json', 'history.json', 'last.pt', 'evaluation.json')):
        raise RuntimeError('Experiment output exists; use --resume or a new output directory')
