"""Kaggle-only preflight and explicitly selected smoke test; never pilot training."""

import argparse
import csv
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight_only', action='store_true')
    parser.add_argument('--download_weights', action='store_true')
    parser.add_argument('--weights', type=Path)
    args = parser.parse_args()
    working = Path('/kaggle/working')
    if not working.is_dir():
        raise RuntimeError('This entrypoint must run inside Kaggle')
    output = working / 'PV-DINOV3/outputs/smoke'
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    # Keep the supplied CUDA torch/torchvision builds intact.
    if metadata.version('timm') != '1.0.29':
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', 'timm==1.0.29'], check=True)
    for name in ['torch', 'torchvision', 'timm', 'huggingface-hub', 'safetensors', 'numpy', 'Pillow']:
        print(f'{name}: {metadata.version(name)}', flush=True)
    with (output / 'core_tests.txt').open('w') as log:
        result = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
                                stdout=log, stderr=subprocess.STDOUT)
    print((output / 'core_tests.txt').read_text(), flush=True)
    result.check_returncode()

    from pvseg.data import discover_root, read_manifest
    from pvseg.runtime import environment, write_json
    from pvseg.model import HF_REPOSITORY, HF_REVISION, INDICES, MODEL_ID
    import inspect
    import timm
    from timm.models.eva import Eva

    directory_counts = []
    with (output / 'file_inventory.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['relative_path', 'size_bytes'])
        for parent, directories, files in os.walk('/kaggle/input'):
            directories.sort()
            directory_counts.append({'path': parent, 'files_here': len(files)})
            for name in sorted(files):
                path = Path(parent) / name
                writer.writerow([str(path.relative_to('/kaggle/input')), path.stat().st_size])
    root = discover_root('/kaggle/input')
    rows = read_manifest(root)
    report = {'environment': environment(), 'data_root': str(root),
              'manifest_pairs': len(rows), 'directories': directory_counts,
              'registered': MODEL_ID in timm.list_models('*dinov3*', pretrained=True, include_tags=True),
              'forward_intermediates_api': str(inspect.signature(Eva.forward_intermediates)),
              'model_id': MODEL_ID, 'indices': list(INDICES),
              'weight_repository': HF_REPOSITORY, 'weight_revision': HF_REVISION,
              'tests_passed': True, 'mode': 'preflight_only' if args.preflight_only else 'smoke',
              'raw_pixels_read_by_preflight': 0}
    write_json(output / 'preflight_report.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.preflight_only:
        print('STOP: preflight only; no weight download, image read, forward or training.', flush=True)
        return
    command = [sys.executable, str(REPO / 'smoke_test_pv896.py'), '--data_root', str(root),
               '--batch_size', '1', '--seed', '42', '--output_dir', str(output)]
    if args.weights:
        command.extend(['--weights', str(args.weights)])
    elif args.download_weights:
        command.append('--download_weights')
    else:
        raise RuntimeError('Specify mounted --weights or explicitly authorize --download_weights')
    subprocess.run(command, check=True)
    print('STOP: smoke complete; pilot training is a separate user decision.', flush=True)


if __name__ == '__main__':
    main()
