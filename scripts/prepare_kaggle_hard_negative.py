"""Package the hard-negative scene-holdout experiment for a private Kaggle run."""

import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile


REPO = Path(__file__).resolve().parents[1]
RUN = Path(r"G:\PV-GF\tmp\hard_negative_rejection_20261101\run")
FILES = (
    "pvseg/hard_negative_gate.py",
    "pvseg/frozen_baselines.py",
    "pvseg/dynamic_brightness_fusion.py",
    "pvseg/losses.py",
    "pvseg/spatial.py",
    "pvseg/local_texture.py",
    "configs/a0a1_selected_20261029.json",
    "scripts/hard_negative_rejection.py",
)


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    source = io.BytesIO()
    manifest = {name: hashlib.sha256((REPO / name).read_bytes()).hexdigest() for name in FILES}
    with zipfile.ZipFile(source, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in FILES:
            archive.write(REPO / name, name)
        archive.writestr("source_sha256.json", json.dumps(manifest, indent=2))
    payload = base64.b64encode(source.getvalue()).decode("ascii")
    launcher = '''import base64, io, pathlib, subprocess, sys, zipfile
root = pathlib.Path('/kaggle/working/hard-negative-source')
root.mkdir(parents=True, exist_ok=True)
payload = "@@PAYLOAD@@"
with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as archive:
    archive.extractall(root)
print('Hard-negative experiment source hashes: @@HASHES@@', flush=True)
subprocess.run([sys.executable, str(root / 'scripts/hard_negative_rejection.py')], cwd=root, check=True)
'''.replace("@@PAYLOAD@@", payload).replace("@@HASHES@@", json.dumps(manifest, sort_keys=True))
    (RUN / "run.py").write_text(launcher, encoding="utf-8")
    metadata = {
        "id": "zh0416/pv-hard-negative-scene-holdout",
        "title": "PV Hard Negative Scene Holdout",
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": ["zh0416/pv-896"],
        "kernel_sources": [
            "zh0416/pv-greenhouse-contrast-pilot/1",
            "zh0416/pv-brightness-dynamic-fusion/1",
        ],
        "model_sources": [],
        "competition_sources": [],
    }
    (RUN / "kernel-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"run": str(RUN), "source_files": manifest,
                      "archive_bytes": len(source.getvalue())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
