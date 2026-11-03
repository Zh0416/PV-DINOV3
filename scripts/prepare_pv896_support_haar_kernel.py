"""Package support-trained dynamic fusion and Haar without old PV weights."""

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile


REPO = Path(__file__).resolve().parents[1]
SOURCE = (REPO.parent / 'tmp/pv896_300_final_review_20261103/'
          'pv896_300_evaluation/results.json')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main(output):
    if output.exists():
        raise FileExistsError(output)
    previous = json.loads(SOURCE.read_text(encoding='utf-8'))
    source = previous['protocol']
    support_id = source['support_ids'][0]
    categories = source['query_ids']
    query_ids = [sample_id for members in categories.values() for sample_id in members]
    if (previous['status'] != 'complete' or len(query_ids) != 300 or
            len(set(query_ids)) != 300 or support_id in query_ids):
        raise ValueError('Locked query selection is incomplete')
    protocol = dict(dataset=source['dataset'], manifest_sha256=source['manifest_sha256'],
                    support_id=support_id, query_ids=categories, steps=80,
                    learning_rate=.002, seed=20261103,
                    foundation='Frozen generic DINOv3-SAT and GeoRSCLIP only',
                    task_parameters='Four fresh dynamic fusion and Haar correction pairs',
                    support_policy='Four checkerboard partitions; fit on three, calibrate on held-out one',
                    threshold_policy='Support-only held-out IoU; no old threshold',
                    presence_policy='True iff predicted mask has at least one pixel',
                    query_label_policy='Read only after all 300 masks are saved',
                    previous_query_use='Query identifiers only; this repeated 300-image evaluation is exploratory',
                    prohibited_assets='No old PV fusion, Haar, brightness, SSPNet checkpoint or calibration')
    paths = [REPO / 'scripts/kaggle_pv896_support_haar_fusion.py',
             REPO / 'pvseg/support_haar_fusion.py',
             REPO / 'pvseg/dinov3_foundation.py',
             REPO / 'pvseg/support_only_segmenter.py',
             REPO / 'pvseg/frequency_gate.py',
             REPO / 'pvseg/hard_negative_gate.py',
             REPO / 'pvseg/frozen_baselines.py',
             REPO / 'pvseg/__init__.py']
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for path in paths:
            bundle.write(path, path.relative_to(REPO).as_posix())
        bundle.writestr('assets/support_haar_protocol.json',
                        json.dumps(protocol, ensure_ascii=False))
    output.mkdir(parents=True)
    bootstrap = (
        "import base64, io, pathlib, subprocess, sys, zipfile\n"
        "root = pathlib.Path('/kaggle/working/pv896-support-haar-source')\n"
        "root.mkdir(parents=True, exist_ok=False)\n"
        f"payload = '{base64.b64encode(archive.getvalue()).decode('ascii')}'\n"
        "with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as bundle:\n"
        "    bundle.extractall(root)\n"
        "subprocess.run([sys.executable, str(root / 'scripts/kaggle_pv896_support_haar_fusion.py')], "
        "cwd=root, check=True)\n")
    (output / 'pv896-support-haar.py').write_text(bootstrap, encoding='ascii')
    metadata = dict(id='zh0416/pv-896-300-support-trained-haar-fusion',
                    title='PV 896 300 Support Trained Haar Fusion',
                    code_file='pv896-support-haar.py', language='python',
                    kernel_type='script', is_private=True, enable_gpu=True,
                    enable_internet=True, dataset_sources=['zh0416/pv-896'],
                    kernel_sources=['zh0416/sspnet-dinov3-pv'],
                    competition_sources=[], model_sources=[])
    (output / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2),
                                                   encoding='utf-8')
    (output / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False,
                                                   indent=2), encoding='utf-8')
    print(json.dumps(dict(kernel=metadata['id'], support=support_id,
                          queries=len(query_ids), archive_bytes=len(archive.getvalue())),
                     ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
