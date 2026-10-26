"""Notebook cell: pin GIT_REF to a reviewed commit before submitting a run."""

from pathlib import Path
import subprocess
import sys


REPOSITORY = 'https://github.com/Zh0416/PV-DINOV3.git'
GIT_REF = 'codex/segdino-sat-smoke'
# Default is an environment/test check. Change only after weights are confirmed.
RUN_ARGUMENTS = ['--preflight_only']

repo = Path('/kaggle/working/PV-DINOV3')
if repo.exists():
    origin = subprocess.check_output(['git', 'remote', 'get-url', 'origin'], cwd=repo, text=True).strip()
    if origin != REPOSITORY:
        raise RuntimeError('Existing checkout has a different remote; preserve it')
    status = subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo, text=True).strip()
    if status:
        raise RuntimeError('Existing checkout has changes; preserve them instead of replacing files')
else:
    subprocess.run(['git', 'clone', '--no-checkout', REPOSITORY, str(repo)], check=True)
subprocess.run(['git', 'fetch', 'origin', GIT_REF], cwd=repo, check=True)
subprocess.run(['git', 'checkout', '--detach', 'FETCH_HEAD'], cwd=repo, check=True)
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
print(f'GitHub commit: {commit}', flush=True)
if len(GIT_REF) == 40 and commit != GIT_REF:
    raise RuntimeError('Resolved commit differs from the requested pinned commit')
subprocess.run([sys.executable, str(repo / 'scripts/kaggle_smoke.py'), *RUN_ARGUMENTS], cwd=repo, check=True)
