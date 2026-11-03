"""Package the dual-prior pilot as a self-contained private Kaggle script."""

import argparse
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile


REPO = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main(output):
    if output.exists():
        raise FileExistsError(output)
    manifest = REPO.parent / 'PV-896/manifest.csv'
    with manifest.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    by_id = {row['sample_id']: row for row in rows}
    support_id = 'pv001183'
    support = next(row for row in rows if row['sample_id'] == support_id)
    queries = []
    train_queries = []
    training_scene = '0812_Q_0a5aa4566e4f48'
    for category in ('正样本', '负样本', '困难正样本'):
        pool = sorted((row for row in rows
                       if row['category'] == category and
                       row['source_scene'] != support['source_scene'] and
                       row['source_image'] != support['source_image'] and
                       row.get('image_sha256') != support.get('image_sha256')),
                      key=lambda row: row['sample_id'])
        queries.extend(row['sample_id'] for row in pool[:8])
        training_pool = sorted((row for row in rows if
                                row['category'] == category and
                                row['source_scene'] == training_scene),
                               key=lambda row: row['sample_id'])
        train_queries.extend(row['sample_id'] for row in training_pool[:2])
    if len(queries) != 24 or support_id in queries:
        raise ValueError('Insufficient disjoint query candidates')
    if (len(train_queries) != 6 or set(train_queries) & set(queries) or
            any(by_id[sample_id]['source_scene'] != '0807_Q_934ae716b5714f'
                for sample_id in queries)):
        raise ValueError('Training and evaluation scenes must be disjoint')
    smoke = [train_queries[index] for index in (0, 2, 4)]
    protocol = dict(dataset='zh0416/pv-896', manifest_sha256=digest(manifest),
                    support_id=support_id, train_query_ids=train_queries,
                    query_ids=queries, smoke_query_ids=smoke, channels=128,
                    smoke_steps=10, ablation_steps=24,
                    learning_rate=.001, weight_decay=.0001,
                    seed=20261103, threshold=.5,
                    query_policy='6 labeled train queries from 0812 scene, 24 development queries from 0807 scene; no evaluation labels used to fit or calibrate',
                    interpretation='Exploratory one-support comparison; these PV-896 scenes have prior development exposure')
    paths = [REPO / name for name in (
        'scripts/kaggle_dual_prior_pilot.py', 'pvseg/dual_prior.py',
        'pvseg/dinov3_foundation.py', 'pvseg/support_haar_fusion.py',
        'pvseg/support_only_segmenter.py', 'pvseg/frequency_gate.py',
        'pvseg/hard_negative_gate.py', 'pvseg/frozen_baselines.py',
        'pvseg/__init__.py')]
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for path in paths:
            bundle.write(path, path.relative_to(REPO).as_posix())
        bundle.writestr('assets/dual_prior_protocol.json',
                        json.dumps(protocol, ensure_ascii=False))
    output.mkdir(parents=True)
    bootstrap = (
        "import base64, io, pathlib, subprocess, sys, zipfile\n"
        "root = pathlib.Path('/kaggle/working/pv-dual-prior-source')\n"
        "root.mkdir(parents=True, exist_ok=False)\n"
        f"payload = '{base64.b64encode(archive.getvalue()).decode('ascii')}'\n"
        "with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as bundle:\n"
        "    bundle.extractall(root)\n"
        "subprocess.run([sys.executable, str(root / 'scripts/kaggle_dual_prior_pilot.py')], "
        "cwd=root, check=True)\n")
    (output / 'pv-dual-prior.py').write_text(bootstrap, encoding='ascii')
    metadata = dict(id='zh0416/pv-dual-prior-feature-fusion-pilot',
                    title='PV Dual Prior Feature Fusion Pilot',
                    code_file='pv-dual-prior.py', language='python',
                    kernel_type='script', is_private=True, enable_gpu=True,
                    enable_internet=True, dataset_sources=['zh0416/pv-896'],
                    kernel_sources=['zh0416/sspnet-dinov3-pv'],
                    competition_sources=[], model_sources=[])
    (output / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2),
                                                   encoding='utf-8')
    (output / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False,
                                                   indent=2), encoding='utf-8')
    print(json.dumps(dict(kernel=metadata['id'], support=support_id,
                          queries=len(queries), smoke=smoke), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
