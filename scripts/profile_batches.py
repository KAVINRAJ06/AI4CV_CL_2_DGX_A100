"""Bounded, process-isolated full-model batch profiling on one CUDA GPU.

Run from the repository root using the training Python environment.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback

import yaml


def choose_batch(results):
    eligible = [r for r in results if r['status'] == 'ok'
                and r['peak_reserved_bytes'] < .85 * r['total_memory_bytes']]
    if not eligible:
        return None
    best = max(r['images_per_second'] for r in eligible)
    return min(r['batch_size'] for r in eligible if r['images_per_second'] >= .95 * best)


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False))


def worker(args):
    import torch
    from torch.nn import functional as F
    from torch.utils.data import Subset
    from tqsi.config import load_config
    from tqsi.data import prepare_manifest, SegmentationDataset
    from tqsi.model import TQSI
    from tqsi.continual import TaskController, segmentation_loss, masked_step
    from tqsi.train import (seed_all, loader, make_replay_memory, unpack_replay,
                           calibrate_binary_threshold, evaluate)
    from tqsi.metrics import Metrics

    cfg = load_config(args.config)
    cfg['batch_size'] = args.batch
    device = torch.device(cfg.get('device', 'cuda'))
    if str(device) == 'cpu':
        raise ValueError('The full-model profiler requires CUDA')
    torch.cuda.set_device(device)
    seed_all(cfg['seed'])
    torch.set_num_threads(cfg.get('cpu_threads', 4))
    tasks = [load_config(p) for p in cfg['tasks']]
    if len({t['name'] for t in tasks}) != len(tasks):
        raise ValueError('Task names must be unique')
    for task in tasks:
        if 'dataset_root' in cfg:
            task['root'] = str(Path(cfg['dataset_root']) / task['relative_root'])
        task['image_size'] = cfg.get('image_size', 512)
        if task.get('num_classes', 1) != cfg['model'].get('num_classes', 1) or task.get('class_names') != tasks[0].get('class_names'):
            raise ValueError('All tasks must share the configured output ontology')
    model = TQSI(cfg['model'], len(tasks)).to(device)
    controller = TaskController()
    replay = []
    params = [p for p in model.parameters() if p.requires_grad]
    bp = [p for p in model.bottleneck.parameters() if p.requires_grad]
    ids = {id(p) for p in bp}
    durations, rows = [], []
    total_images = 0
    torch.cuda.reset_peak_memory_stats(device)
    for task_id, task in enumerate(tasks):
        seed_all(cfg['seed'] + task_id)
        print(f"batch={args.batch}, task={task['name']}: preparing real tiles", flush=True)
        manifest = prepare_manifest(task, Path(args.output) / 'splits', cfg.get('split_seed', 42))
        train = SegmentationDataset(task, manifest['splits']['train'], augment=True,
                                    limit=cfg.get('max_samples', {}).get('train'), sampling=cfg.get('train_sampling'))
        if len(train) < args.batch:
            raise ValueError(f"Only {len(train)} training tiles; cannot measure full batch {args.batch}")
        sampler = train.training_sampler(cfg['seed'] + task_id)
        batches = loader(train, cfg, True, sampler, cfg['seed'] + task_id)
        references = SegmentationDataset(task, manifest['splits']['train'], limit=cfg.get('reference_samples', 8))
        with torch.no_grad():
            images = torch.stack([references[i][0] for i in range(len(references))]).to(device)
            features = torch.cat([model.backbone.encode(x) for x in images.split(args.batch)])
            controller.register(task_id, F.normalize(model.projection(features), dim=-1))
        del images, features
        optimizer = torch.optim.AdamW([
            {'params': [p for p in params if id(p) not in ids], 'lr': cfg['lr']},
            {'params': bp, 'lr': cfg.get('bottleneck_lr', cfg['lr'])},
        ], weight_decay=cfg.get('weight_decay', 0.))
        iterator = iter(batches)
        model.train()
        replay_cfg = cfg.get('replay', {})
        for step in range(args.warmup + args.steps):
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            try:
                images, targets = next(iterator)
            except StopIteration:
                iterator = iter(batches)
                images, targets = next(iterator)
            if len(images) != args.batch:
                iterator = iter(batches)
                images, targets = next(iterator)
            images, targets = images.to(device), targets.to(device)
            replay_target = teacher = None
            if replay and replay_cfg.get('batch_size', 0):
                indices = torch.randint(len(replay), (replay_cfg['batch_size'],)).tolist()
                rx, replay_target, teacher = unpack_replay([replay[i] for i in indices], device)
                images = torch.cat((images, rx))
            logits = model(images)
            seg = segmentation_loss(logits[:args.batch], targets, cfg.get('loss'))
            rehearsal = segmentation_loss(logits[args.batch:], replay_target, cfg.get('loss')) if replay_target is not None else seg * 0
            distill = F.mse_loss(logits[args.batch:], teacher) if teacher is not None else seg * 0
            sep, stab = controller.losses(model.bottleneck, cfg.get('lambda_sep', .1) != 0, cfg.get('lambda_stab', .1) != 0)
            loss = (seg + cfg.get('lambda_sep', .1)*sep + cfg.get('lambda_stab', .1)*stab
                    + replay_cfg.get('weight', 1.)*rehearsal + replay_cfg.get('distill_weight', 0.)*distill)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            # masked_step checks the global gradient norm before every optimizer update.
            masked_step(loss, optimizer, model.bottleneck.masked_parameters(task_id), params, cfg.get('grad_clip', 1.))
            metrics = Metrics(model.classes)
            metrics.update(logits[:args.batch].detach(), targets)
            # Match the scalar synchronization in the production training loop.
            loss_value = float(loss.detach())
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            print(f"batch={args.batch}, task={task_id}, step={step+1}: {elapsed:.3f}s loss={loss_value:.5f}", flush=True)
            if step >= args.warmup:
                durations.append(elapsed)
                total_images += args.batch
            del logits, loss, seg, rehearsal, distill, sep, stab, images, targets
        if args.smoke:
            val = SegmentationDataset(task, manifest['splits']['val'], augment=False,
                                      limit=cfg.get('max_samples', {}).get('val'), sampling=cfg.get('bounded_evaluation_sampling'))
            bounded = Subset(val, range(min(len(val), args.batch * 2)))
            if not len(bounded):
                raise ValueError('Empty validation split')
            validation = loader(bounded, cfg)
            threshold = calibrate_binary_threshold(model, validation, device)[0] if model.classes == 1 else 0.
            result = evaluate(model, validation, device, cfg.get('loss'), threshold)
            if not math.isfinite(result['loss']):
                raise ValueError('Nonfinite validation loss')
            rows.append({'task': task['name'], 'validation_loss': result['loss'], 'validation_images': len(bounded)})
        controller.freeze(task_id, model.bottleneck)
        if replay_cfg.get('samples_per_task', 0) and task_id + 1 < len(tasks):
            memory = SegmentationDataset(task, manifest['splits']['train'], limit=replay_cfg['samples_per_task'])
            replay.extend(make_replay_memory(model, memory, replay_cfg['samples_per_task'], device, args.batch))
        del optimizer, iterator, batches
    return dict(status='ok', batch_size=args.batch, seconds_per_step=sum(durations)/len(durations),
                images_per_second=total_images/sum(durations), measured_steps=len(durations),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                total_memory_bytes=torch.cuda.get_device_properties(device).total_memory,
                gpu=torch.cuda.get_device_name(device), torch_version=torch.__version__, validation=rows)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True, help='New directory for profiling artifacts')
    parser.add_argument('--batches', nargs='+', type=int, default=[4, 8, 12, 16])
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--timeout', type=int, default=1800, help='Maximum seconds per candidate process')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--batch', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--smoke', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.batches) < 1 or args.warmup < 0 or args.steps < 1 or args.timeout < 1:
        parser.error('Positive batches, steps and timeout, and nonnegative warmup required')
    out = Path(args.output).resolve()
    if args.worker:
        try:
            result = worker(args)
        except Exception as exc:
            import torch
            result = dict(status='oom' if isinstance(exc, torch.cuda.OutOfMemoryError) else 'error',
                          batch_size=args.batch, error=str(exc), traceback=traceback.format_exc())
            print(result['traceback'], flush=True)
        save_json(out / 'result.json', result)
        return
    from tqsi.config import load_config
    cfg = load_config(args.config)
    if cfg.get('device', 'auto') == 'auto':
        cfg['device'] = 'cuda:0'
    if cfg.get('device') == 'cpu':
        parser.error('Use the actual CUDA training configuration')
    if out == Path(cfg['output_dir']).resolve() or Path(cfg['output_dir']).resolve() in out.parents:
        parser.error('Profiler output must be separate from training output')
    out.mkdir(parents=True, exist_ok=False)
    snapshot = out / 'input_config.yaml'
    snapshot.write_text(yaml.safe_dump(cfg, sort_keys=False))
    def launch(batch, smoke=False):
        directory = out / (f'smoke_{batch}' if smoke else f'batch_{batch}')
        directory.mkdir()
        command = [sys.executable, '-u', '-m', 'scripts.profile_batches', '--worker', '--config', str(snapshot),
                   '--output', str(directory), '--batch', str(batch), '--warmup', str(args.warmup),
                   '--steps', str(3 if smoke else args.steps)]
        if smoke:
            command.append('--smoke')
        print(f"Starting {'smoke' if smoke else 'profile'} batch {batch}; log: {directory / 'worker.log'}", flush=True)
        with (directory / 'worker.log').open('w') as log:
            try:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
            except subprocess.TimeoutExpired:
                return dict(status='timeout', batch_size=batch)
        result_path = directory / 'result.json'
        if completed.returncode != 0 or not result_path.exists():
            return dict(status='error', batch_size=batch, returncode=completed.returncode)
        return json.loads(result_path.read_text())
    results = []
    report = dict(results=results, recommendation=None,
                  note='Short training trajectories with real replay tiles and teacher logits; not a quality benchmark. Throughput counts new-task images and includes data loading and metrics. Run on an otherwise idle GPU.')
    for batch in sorted(set(args.batches)):
        result = launch(batch)
        results.append(result)
        save_json(out / 'report.json', report)
        print(json.dumps(result), flush=True)
        if result['status'] != 'ok':
            break
    selected = choose_batch(results)
    if selected is not None:
        smoke = launch(selected, True)
        report['smoke'] = smoke
        if smoke['status'] == 'ok' and smoke['peak_reserved_bytes'] < .85 * smoke['total_memory_bytes']:
            recommended = copy.deepcopy(cfg)
            recommended['batch_size'] = selected
            recommended['output_dir'] = str(out / 'recommended_training')
            (out / 'recommended.yaml').write_text(yaml.safe_dump(recommended, sort_keys=False))
            report['recommendation'] = selected
    save_json(out / 'report.json', report)
    print(f"Report: {out / 'report.json'}; recommended batch: {report['recommendation']}", flush=True)
    if report['recommendation'] is None:
        raise SystemExit('No validated recommendation; inspect worker logs')


if __name__ == '__main__':
    main()
