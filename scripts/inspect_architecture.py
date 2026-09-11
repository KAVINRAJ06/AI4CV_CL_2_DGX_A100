"""Count the actual configured model; optionally time synthetic batches on an idle GPU.

This does not load training state, update weights, or measure dataset I/O.
"""
import argparse
import json
import time
import statistics
from pathlib import Path

import torch

from tqsi.config import load_config
from tqsi.continual import segmentation_loss
from tqsi.metrics import Metrics
from tqsi.model import TQSI


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', required=True, help='Training YAML or resolved_config.json')
    parser.add_argument('--output', required=True, help='New JSON file')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--batch-size', type=int, help='Override batch size for bounded profiling')
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 0:
        parser.error('steps must be positive and warmup nonnegative')
    output = Path(args.output)
    if output.exists():
        parser.error('Choose a new output file')
    cfg = load_config(args.config)
    if args.batch_size is not None:
        if args.batch_size < 1:
            parser.error('batch-size must be positive')
        cfg['batch_size'] = args.batch_size
    torch.set_num_threads(cfg.get('cpu_threads', 4))
    model = TQSI(cfg['model'], len(cfg['tasks']))
    counts = {name: {'total': sum(p.numel() for p in module.parameters()),
                     'trainable': sum(p.numel() for p in module.parameters() if p.requires_grad)}
              for name, module in model.named_children()}
    report = {'config': cfg['model'], 'modules': counts,
              'total_parameters': sum(p.numel() for p in model.parameters()),
              'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
              'active_mask_parameters_per_task': [sum(int(mask.sum()) for _, mask in model.bottleneck.masked_parameters(t))
                                                  for t in range(len(cfg['tasks']))]}
    if args.profile:
        device = torch.device(args.device)
        if device.type != 'cuda':
            parser.error('Use an idle CUDA GPU for timing')
        model.to(device).train()
        batch, size = cfg['batch_size'], cfg['image_size']
        x = torch.rand(batch, 3, size, size, device=device)
        y = (torch.rand(batch, size, size, device=device) < .01).long()
        measured = {}
        current = {}

        def timed(name, function):
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            result = function()
            torch.cuda.synchronize(device)
            current[name] = current.get(name, 0.) + time.perf_counter() - start
            return result

        # These regions partition the forward path; nested head regions are not summed twice.
        for module, method, name in [(model.backbone, 'encode', 'sam_encoder'),
                                     (model.bottleneck, 'forward', 'quantum_or_control')]:
            original = getattr(module, method)
            def wrapped(*values, _original=original, _name=name, **kwargs):
                return timed(_name, lambda: _original(*values, **kwargs))
            setattr(module, method, wrapped)
        for name in ('feature_adapter', 'projection', 'spatial_decoder', 'decoder_head', 'spatial_decoder_head'):
            module = getattr(model, name, None)
            if module is not None:
                original = module.forward
                def wrapped(*values, _original=original, _name=name, **kwargs):
                    return timed(_name, lambda: _original(*values, **kwargs))
                module.forward = wrapped
        if model.decoder_mode == 'sam_prompt':
            original_decode = model.backbone.decode
            model.backbone.decode = lambda *values: timed('sam_decode', lambda: original_decode(*values))
        if model.spatial_decoder is not None:
            for name, module in model.spatial_decoder.named_children():
                original = module.forward
                def wrapped(*values, _original=original, _name='decoder_detail.'+name, **kwargs):
                    return timed(_name, lambda: _original(*values, **kwargs))
                module.forward = wrapped
        for mode in ('train', 'eval'):
            model.train(mode == 'train')
            for step in range(args.warmup + args.steps):
                current = {}
                model.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(mode == 'train'):
                    logits = timed('full_forward', lambda: model(x))
                    loss = timed('segmentation_loss', lambda: segmentation_loss(logits, y, cfg.get('loss')))
                if mode == 'train':
                    timed('backward', loss.backward)
                    if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                        raise ValueError('Nonfinite gradients')
                timed('metrics', lambda: Metrics(model.classes).update(logits.detach(), y))
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite loss')
                if step >= args.warmup:
                    for name, seconds in current.items():
                        measured.setdefault(mode+'.'+name, []).append(seconds)
                print(f'{mode} step {step+1}/{args.warmup+args.steps}: forward={current["full_forward"]:.3f}s backward={current.get("backward",0):.3f}s', flush=True)
                del logits, loss
        report['synthetic_profile'] = {'gpu': torch.cuda.get_device_name(device), 'batch_size': batch,
            'torch': torch.__version__, 'image_size': size, 'measured_steps_per_mode': args.steps,
            'warmup_steps_per_mode': args.warmup,
            'mean_seconds': {k: sum(v)/len(v) for k, v in measured.items()},
            'median_seconds': {k: statistics.median(v) for k, v in measured.items()},
            'samples_seconds': measured,
            'limitations': 'Random inputs and initialized trainable weights; synchronized regions add overhead. full_forward includes all module timings; decoder_detail rows are nested inside spatial_decoder and must not be added again. Excludes dataset I/O, transfer, optimizer, replay, auxiliary losses, calibration and checkpointing. Not an epoch prediction.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
