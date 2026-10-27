"""Kaggle script template: substitute a reviewed 40-character commit before push."""

from pathlib import Path
import subprocess
import sys

GIT_REF = 'REPLACE_WITH_REVIEWED_40_CHARACTER_COMMIT'
REPOSITORY = 'https://github.com/Zh0416/PV-DINOV3.git'

if len(GIT_REF) != 40 or any(c not in '0123456789abcdef' for c in GIT_REF):
    raise RuntimeError('Pin this launch to a full Git commit before submitting')
repo = Path('/kaggle/working/PV-SSPNET')
if repo.exists():
    raise RuntimeError('Existing checkout must be preserved; choose a fresh Kaggle run')
subprocess.run(['git', 'clone', '--no-checkout', REPOSITORY, str(repo)], check=True)
subprocess.run(['git', 'fetch', 'origin', GIT_REF], cwd=repo, check=True)
subprocess.run(['git', 'checkout', '--detach', GIT_REF], cwd=repo, check=True)
actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
if actual != GIT_REF:
    raise RuntimeError('Pinned commit mismatch')
print(f'SSPNet source commit: {actual}', flush=True)
subprocess.run([sys.executable, str(repo/'scripts/kaggle_sspnet.py')], cwd=repo, check=True)
