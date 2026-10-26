"""Run only the audited <=12 pairs, one optimizer step, then stop."""

import argparse
import json
from pathlib import Path
import time
import traceback

import torch
from torch.utils.data import DataLoader

from pvseg.data import PV896Dataset, discover_root, read_manifest, select_audit_rows
from pvseg.losses import binary_loss
from pvseg.model import FrozenSATSegDINO, resolve_weights, sha256_file
from pvseg.runtime import (Precision, decoder_optimizer, environment, load_checkpoint, memory,
                           save_checkpoint, save_prediction, seed_everything, sync, timed, write_json)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--data_root', type=Path)
    result.add_argument('--input_root', type=Path, default=Path('/kaggle/input'))
    result.add_argument('--image_dir', type=Path)
    result.add_argument('--mask_dir', type=Path)
    result.add_argument('--weights', type=Path)
    result.add_argument('--download_weights', action='store_true')
    result.add_argument('--weight_cache', type=Path, default=Path('/kaggle/working/model_cache'))
    result.add_argument('--output_dir', type=Path, default=Path('/kaggle/working/PV-DINOV3/outputs/smoke'))
    result.add_argument('--img_size', type=int, default=896)
    result.add_argument('--batch_size', type=int, default=1)
    result.add_argument('--seed', type=int, default=42)
    result.add_argument('--sample_ids', type=Path, default=Path(__file__).parent / 'configs/smoke_sample_ids.json')
    return result


def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {'status': 'RUNNING', 'checks': {}, 'training_steps': 0,
              'purpose': 'Engineering smoke test; not a research accuracy evaluation',
              'config': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    report_path = args.output_dir / 'smoke_report.json'
    stage = 'configuration'
    start = time.perf_counter()

    def check(name, details):
        report['checks'][name] = {'pass': True, 'details': details}
        write_json(report_path, report)
        print(f'PASS {name}: {details}', flush=True)

    try:
        if args.batch_size != 1 or args.img_size != 896:
            raise ValueError('First smoke run requires batch_size=1 and img_size=896')
        seed_everything(args.seed)
        report['environment'] = environment()
        if not torch.cuda.is_available():
            raise RuntimeError('No CUDA GPU allocated; do not silently substitute a CPU smoke test')
        device = torch.device('cuda:0')
        precision = Precision(device)
        report['precision'] = precision.describe()
        root = args.data_root or discover_root(args.input_root)
        all_rows = read_manifest(root)
        selected = select_audit_rows(all_rows, args.seed)
        expected_ids = json.loads(args.sample_ids.read_text())['sample_ids']
        if [r['sample_id'] for r in selected] != expected_ids or len(selected) > 12:
            raise ValueError('Selected IDs differ from the already audited <=12 pairs')
        write_json(args.output_dir / 'selected_samples.json', selected)
        report['data'] = {'root': str(root), 'manifest_sha256': sha256_file(root / 'manifest.csv'),
                          'sample_ids': expected_ids, 'unique_pairs': len(selected),
                          'uncertain_source_pairs': sum(r['source_label_uncertain_filename'].lower() == 'true' for r in selected)}
        stage = 'pretrained_weights'
        (path, config, source), seconds = timed(lambda: resolve_weights(
            args.weights, args.download_weights, str(args.weight_cache)))
        report['weight_acquisition_seconds'] = seconds
        report['weight_source'] = source
        stage = 'model_construction'
        model, seconds = timed(lambda: FrozenSATSegDINO(path, config, source, args.img_size).to(device))
        report['model_load_seconds'] = seconds
        report['model'] = model.metadata
        report['parameters'] = {'backbone': sum(p.numel() for p in model.backbone.parameters()),
                                'decoder': sum(p.numel() for p in model.decoder.parameters())}
        check('weights_and_model_api', model.metadata)
        stage = 'data_loader'
        normalization = model.metadata['normalization']
        dataset = PV896Dataset(root, selected, **normalization, img_size=args.img_size,
                               image_dir=args.image_dir, mask_dir=args.mask_dir)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        observed = []
        for batch in loader:
            if (batch['image'].shape != (1, 3, 896, 896) or batch['mask'].shape != (1, 1, 896, 896)
                    or batch['image'].dtype != torch.float32 or batch['mask'].dtype != torch.float32):
                raise AssertionError('Unexpected loader shape/dtype')
            observed.append({'id': batch['sample_id'][0], 'category': batch['category'][0],
                             'foreground_ratio': batch['mask'].mean().item(),
                             'image_shape': list(batch['image'].shape), 'mask_shape': list(batch['mask'].shape),
                             'dtype': str(batch['image'].dtype)})
        check('data_loader_all_selected', observed)
        batch = next(iter(loader))
        images, target = batch['image'].to(device), batch['mask'].to(device)
        optimizer = decoder_optimizer(model)
        model.train()
        if model.backbone.training or any(m.training for m in model.backbone.modules()):
            raise AssertionError('Backbone must remain eval after model.train()')
        before_versions = [p._version for p in model.backbone.parameters()]
        decoder_before = {n: p.detach().clone() for n, p in model.decoder.named_parameters()}
        check('frozen_encoder_and_optimizer', {'backbone_eval': True, 'decoder_only_optimizer': True})
        torch.cuda.reset_peak_memory_stats()
        stage = 'backbone_forward'
        sync()
        step_start = time.perf_counter()
        with precision.context():
            maps, seconds = timed(lambda: model.extract_features(images))
            report['backbone_forward_seconds'] = seconds
            shapes = [{'shape': list(x.shape), 'dtype': str(x.dtype), 'requires_grad': x.requires_grad} for x in maps]
            if any(x.requires_grad or not torch.isfinite(x).all() for x in maps):
                raise AssertionError('Frozen feature maps must be finite and detached')
            report['backbone_forward_memory'] = memory()
            check('four_backbone_features', shapes)
            stage = 'decoder_forward'
            logits, decoder_seconds = timed(lambda: model.decoder(maps, images.shape[-2:]))
            report['decoder_forward_seconds'] = decoder_seconds
            report['batch_forward_seconds'] = seconds + decoder_seconds
            if logits.shape != target.shape:
                raise AssertionError('Decoder shape mismatch')
            check('decoder_output', {'shape': list(logits.shape), 'dtype': str(logits.dtype)})
            loss, loss_seconds = timed(lambda: binary_loss(logits, target))
        if not torch.isfinite(loss):
            raise AssertionError('Nonfinite loss')
        check('finite_loss', float(loss.detach()))
        stage = 'backward_and_optimizer'
        sync()
        backward_start = time.perf_counter()
        precision.scaler.scale(loss).backward()
        precision.scaler.unscale_(optimizer)
        grads = [p.grad for p in model.decoder.parameters()]
        if any(g is None or not torch.isfinite(g).all() for g in grads):
            raise AssertionError('Missing/nonfinite decoder gradients')
        if any(p.grad is not None for p in model.backbone.parameters()):
            raise AssertionError('Frozen backbone acquired gradients')
        grad_norm = sum(float(g.float().square().sum()) for g in grads) ** .5
        if not grad_norm > 0:
            raise AssertionError('Zero decoder gradient norm')
        precision.scaler.step(optimizer)
        precision.scaler.update()
        sync()
        report['backward_optimizer_seconds'] = time.perf_counter() - backward_start
        report['training_step_seconds'] = report['batch_forward_seconds'] + loss_seconds + report['backward_optimizer_seconds']
        report['instrumented_step_wall_seconds'] = time.perf_counter() - step_start
        report['training_steps'] = 1
        changed = [n for n, p in model.decoder.named_parameters() if not torch.equal(p, decoder_before[n])]
        if not changed or before_versions != [p._version for p in model.backbone.parameters()]:
            raise AssertionError('Decoder did not update or backbone was modified')
        report['training_peak_memory'] = memory()
        check('backward_optimizer_step', {'decoder_grad_norm': grad_norm, 'changed_decoder_tensors': len(changed),
                                          'backbone_gradients': 0, 'backbone_parameter_versions_unchanged': True})
        optimizer.zero_grad(set_to_none=True)
        del maps, logits, loss, decoder_before
        model.eval()
        stage = 'checkpoint_roundtrip'
        with torch.no_grad(), precision.context():
            baseline = model(images).detach().clone()
        checkpoint = args.output_dir / 'smoke_decoder.pt'
        save_checkpoint(checkpoint, model, optimizer, report['config'], report['environment'],
                        selected_sample_ids=expected_ids, training_steps=1)
        with torch.no_grad():
            next(model.decoder.parameters()).add_(.125)
        restored_optimizer = decoder_optimizer(model)
        load_checkpoint(checkpoint, model, restored_optimizer)
        with torch.no_grad(), precision.context():
            restored = model(images)
        torch.testing.assert_close(restored, baseline, rtol=1e-4, atol=1e-5)
        check('checkpoint_reload', {'path': str(checkpoint),
                                    'max_abs_prediction_difference': float((restored - baseline).abs().max()),
                                    'rtol': 1e-4, 'atol': 1e-5,
                                    'scope': 'Decoder and optimizer restored; verified pretrained frozen encoder reused'})
        del baseline, restored, images, target
        stage = 'all_sample_predictions'
        predictions = []
        for batch in loader:
            with torch.no_grad(), precision.context():
                logits = model(batch['image'].to(device))
                loss = binary_loss(logits, batch['mask'].to(device))
            if not torch.isfinite(loss) or not torch.isfinite(logits).all():
                raise AssertionError('Nonfinite selected-sample prediction/loss')
            save_prediction(args.output_dir / 'predictions', batch, logits, normalization)
            predictions.append({'id': batch['sample_id'][0], 'category': batch['category'][0],
                                'loss': float(loss), 'prediction_foreground_ratio': float((logits.sigmoid() >= .5).float().mean())})
        check('all_sample_finite_predictions', predictions)
        report['final_memory'] = memory()
        report['elapsed_seconds'] = time.perf_counter() - start
        report['status'] = 'PASS'
        report['stop_reason'] = 'Smoke test complete. No pilot/full training authorized in this run.'
        write_json(report_path, report)
        print(json.dumps(report, indent=2, default=str), flush=True)
    except Exception as exc:
        report.update(status='FAIL', failed_stage=stage, error=f'{type(exc).__name__}: {exc}',
                      traceback=traceback.format_exc(), elapsed_seconds=time.perf_counter() - start)
        if torch.cuda.is_available():
            report['failure_memory'] = memory()
        write_json(report_path, report)
        raise


if __name__ == '__main__':
    run(parser().parse_args())
