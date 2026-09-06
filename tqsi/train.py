"""Notebook/CLI trainer. torchrun uses DDP for gradient synchronization."""
import argparse
import copy
import os
from pathlib import Path
import random
import time
import warnings
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from .artifacts import write_json, history_artifacts, predictions, tsne
from .config import load_config
from .data import prepare_manifest, SegmentationDataset
from .model import TQSI
from .continual import TaskController, segmentation_loss, masked_step
from .metrics import Metrics, forgetting


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def loader(dataset, cfg, train=False, sampler=None, seed=42):
    workers = int(cfg.get("workers", 0))
    return DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=train and sampler is None,
                      sampler=sampler, num_workers=workers, pin_memory=torch.cuda.is_available(),
                      worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(seed),
                      persistent_workers=workers > 0)


@torch.no_grad()
def calibrate_binary_threshold(model, batches, device, quantiles=257):
    """Select the binary Dice threshold on validation only; never use test labels."""
    logits, targets = [], []
    model.eval()
    for images, target in batches:
        logits.append(model(images.to(device))[:, 0].flatten().cpu())
        targets.append(target.flatten().cpu())
    logits, targets = torch.cat(logits), torch.cat(targets)
    valid = targets != -100
    logits, targets = logits[valid], targets[valid].bool()
    if not targets.any():
        return 0.0, float("nan")
    candidates = torch.quantile(logits, torch.linspace(0, 1, quantiles))
    candidates = torch.unique(torch.cat((candidates, torch.tensor([0.]))))
    truth = targets.sum()
    dice = torch.stack([2*((logits > threshold) & targets).sum().float()/((logits > threshold).sum()+truth).clamp_min(1) for threshold in candidates])
    best = int(dice.argmax())
    return float(candidates[best]), float(dice[best])


@torch.no_grad()
def evaluate(model, batches, device, loss_config=None, threshold=0.0):
    model.eval()
    metrics = Metrics(model.classes)
    total, count = 0., 0
    for images, targets in batches:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        total += float(segmentation_loss(logits, targets, loss_config))*len(images)
        count += len(images)
        metrics.update(logits, targets, threshold)
    if not count:
        raise ValueError("Empty evaluation split")
    result = metrics.compute()
    result["loss"] = total/count
    return result


def save_checkpoint(path, state):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def selection_score(metrics):
    """Prefer defined Dice; if a bounded split is all-empty, minimize val loss."""
    value = metrics["dice"]
    if np.isfinite(value):
        return (1, value)
    if not np.isfinite(metrics["loss"]):
        raise ValueError("Validation Dice and loss are both undefined")
    return (0, -metrics["loss"])


def load_checkpoint(path, device="cpu"):
    # Load only trusted checkpoints created by this project (contains RNG state).
    return torch.load(path, map_location=device, weights_only=False)


@torch.no_grad()
def router_report(model, prototypes, loaders, device):
    """Diagnostic task classification; task ID does not alter specified forward."""
    ids = sorted(prototypes)
    bank = torch.stack([prototypes[t] for t in ids]).to(device)
    bank = torch.nn.functional.normalize(bank, dim=-1)
    correct, count = 0, 0
    confusion = np.zeros((len(ids), len(ids)), dtype=int)
    for task, batches in enumerate(loaders):
        for images, _ in batches:
            z = model.backbone.encode(images.to(device)).mean((-2, -1))
            predicted = (torch.nn.functional.normalize(z, dim=-1) @ bank.T).argmax(1).cpu()
            correct += int((predicted == task).sum())
            count += len(images)
            for p in predicted.tolist():
                confusion[task, p] += 1
    return dict(task_accuracy=correct/count, confusion=confusion.tolist(),
                segmentation_routing="oracle and router logits identical: task masks affect training only")


def run(config, resume=None):
    cfg = load_config(config) if isinstance(config, (str, Path)) else copy.deepcopy(config)
    for key in ("batch_size", "epochs", "image_size", "reference_samples"):
        if int(cfg.get(key, 1)) < 1:
            raise ValueError(f"{key} must be positive")
    if any(int(value) < 1 for value in cfg.get("max_samples", {}).values()):
        raise ValueError("max_samples limits must be positive")
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("DGX DDP requires CUDA")
        torch.cuda.set_device(local)
        dist.init_process_group("nccl")
    device_name = cfg.get("device", "auto")
    device = torch.device(f"cuda:{local}" if distributed else ("cuda" if torch.cuda.is_available() else "cpu") if device_name == "auto" else device_name)
    seed_all(cfg["seed"])
    torch.set_num_threads(cfg.get("cpu_threads", 4))
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    if not resume and (out / "resolved_config.json").exists():
        raise FileExistsError(f"Run already exists at {out}; use resume or choose a fresh output_dir")
    tasks = [load_config(path) for path in cfg["tasks"]]
    if len({t["name"] for t in tasks}) != len(tasks):
        raise ValueError("Task names must be unique")
    classes = cfg["model"].get("num_classes", 1)
    for task in tasks:
        if task.get("num_classes", 1) != classes or task["class_names"] != tasks[0]["class_names"]:
            raise ValueError("A sequence must share num_classes and semantic class_names; map labels in dataset YAML")
        if "dataset_root" in cfg:
            task["root"] = str(Path(cfg["dataset_root"]) / task["relative_root"])
        task["image_size"] = cfg.get("image_size", task.get("image_size", 256))
    manifest_dir = out / "splits"
    if rank == 0:
        for task in tasks:
            prepare_manifest(task, manifest_dir, cfg.get("split_seed", 42))
    if distributed:
        dist.barrier()
    import json
    manifests = [json.loads((manifest_dir / f"{task['name']}.json").read_text()) for task in tasks]
    datasets = [{s: SegmentationDataset(
        task, manifest["splits"][s], augment=s == "train", limit=cfg.get("max_samples", {}).get(s),
        sampling=cfg.get("train_sampling") if s == "train" else cfg.get("bounded_evaluation_sampling"),
    ) for s in ("train", "val", "test")} for task, manifest in zip(tasks, manifests)]
    if rank == 0:
        print("Split sizes (source images -> tiles used):", flush=True)
        for task, manifest, sets in zip(tasks, manifests, datasets):
            print(task["name"], {s: (len(manifest["splits"][s]), len(sets[s])) for s in sets}, flush=True)
            if any(manifest["audit"].values()):
                print("  Omitted unpaired files:", {k: len(v) for k, v in manifest["audit"].items()}, flush=True)
    model = TQSI(cfg["model"], len(tasks)).to(device)
    controller = TaskController()
    history, diagnostics, matrix, replay, prototypes, decision_thresholds = [], [], [], [], {}, {}
    start_task = 0
    if resume:
        checkpoint = load_checkpoint(resume, device)
        old = checkpoint["config"]
        for key in ("model", "tasks", "seed", "split_seed", "image_size", "max_samples", "replay", "lr", "epochs", "lambda_sep", "lambda_stab"):
            if old.get(key) != cfg.get(key):
                raise ValueError(f"Resume configuration differs at {key}")
        if checkpoint["fingerprints"] != [m["fingerprint"] for m in manifests]:
            raise ValueError("Resume dataset fingerprints differ")
        model.load_state_dict(checkpoint["model"])
        controller.load_state_dict(checkpoint["controller"])
        history, diagnostics, matrix = checkpoint["history"], checkpoint["diagnostics"], checkpoint["matrix"]
        replay = [(x.cpu(), y.cpu()) for x, y in checkpoint["replay"]]
        prototypes = {t: p.cpu() for t, p in checkpoint["prototypes"].items()}
        decision_thresholds = checkpoint.get("decision_thresholds", {})
        start_task = checkpoint["completed_task"]+1
    params = [p for p in model.parameters() if p.requires_grad]
    if rank == 0:
        write_json(out / "resolved_config.json", cfg)
        write_json(out / "datasets.json", tasks)
        write_json(out / "environment.json", dict(torch=torch.__version__, device=str(device), world_size=world,
            trainable_parameters=sum(p.numel() for p in params), bottleneck_parameters=sum(p.numel() for p in model.bottleneck.parameters()),
            backbone=cfg["model"]["backbone"], bounded_smoke=bool(cfg.get("max_samples"))))
        if cfg["model"]["bottleneck_type"] in ("quantum", "classical_orthogonal") and cfg.get("lambda_sep", .1):
            warnings.warn("Shared unitary/orthogonal transforms preserve overlap. L_sep is a diagnostic constant for fixed inputs; see docs/SPEC_REVIEW.md")
    wrapped = DDP(model, device_ids=[local], broadcast_buffers=False) if distributed else model
    diagnostic_split = cfg.get("diagnostic_evaluation_split")
    if diagnostic_split not in (None, "train"):
        raise ValueError("diagnostic_evaluation_split may only be 'train' or omitted")
    if diagnostic_split == "train":
        # Capacity diagnostic only. Never use these selected weights as a
        # generalization claim or final checkpoint for a benchmark.
        val_datasets = [SegmentationDataset(
            task, manifest["splits"]["train"], augment=False,
            limit=cfg.get("max_samples", {}).get("train"), sampling=cfg.get("train_sampling"),
        ) for task, manifest in zip(tasks, manifests)]
    else:
        val_datasets = [d["val"] for d in datasets]
    val_loaders = [loader(d, cfg) for d in val_datasets]
    test_loaders = [loader(d["test"], cfg) for d in datasets]
    for task_id in range(start_task, len(tasks)):
        seed_all(cfg["seed"]+task_id)
        task_name = tasks[task_id]["name"]
        sampler = datasets[task_id]["train"].training_sampler(cfg["seed"] + task_id, world, rank)
        if sampler is None and distributed:
            sampler = DistributedSampler(datasets[task_id]["train"], shuffle=True, seed=cfg["seed"])
        train_batches = loader(datasets[task_id]["train"], cfg, True, sampler, seed=cfg["seed"]+task_id)
        reference_dataset = SegmentationDataset(tasks[task_id], manifests[task_id]["splits"]["train"], limit=cfg.get("reference_samples", 8))
        reference_x = torch.stack([reference_dataset[i][0] for i in range(len(reference_dataset))]).to(device)
        with torch.no_grad():
            # Chunk encoder work to the configured batch size, important on 4GB GPUs.
            ref_z = torch.cat([model.backbone.encode(x) for x in reference_x.split(cfg["batch_size"])])
            controller.register(task_id, torch.nn.functional.normalize(model.projection(ref_z), dim=-1))
            prototypes[task_id] = ref_z.mean((0, 2, 3)).cpu()
        del reference_x, ref_z
        # Reset moments at each task boundary; resume follows the same policy.
        optimizer = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"], eta_min=cfg["lr"]*.01)
        best = (-1, -float("inf"))
        for epoch in range(cfg["epochs"]):
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            model.train()
            metric = Metrics(classes)
            total, count, sums = 0., 0, [0., 0., 0.]
            begin = time.perf_counter()
            for images, targets in train_batches:
                images, targets = images.to(device), targets.to(device)
                n = len(images)
                replay_target = None
                if replay and cfg.get("replay", {}).get("batch_size", 0):
                    indices = torch.randint(len(replay), (cfg["replay"]["batch_size"],)).tolist()
                    rx = torch.stack([replay[i][0] for i in indices]).to(device)
                    replay_target = torch.stack([replay[i][1] for i in indices]).to(device)
                    images = torch.cat((images, rx))
                logits = wrapped(images)
                seg = segmentation_loss(logits[:n], targets, cfg.get("loss"))
                replay_loss = segmentation_loss(logits[n:], replay_target, cfg.get("loss")) if replay_target is not None else seg*0
                sep, stab = controller.losses(model.bottleneck)
                loss = seg + cfg.get("lambda_sep", .1)*sep + cfg.get("lambda_stab", .1)*stab + cfg.get("replay", {}).get("weight", 1.)*replay_loss
                masked_step(loss, optimizer, model.bottleneck.masked_parameters(task_id), params, cfg.get("grad_clip", 1.))
                metric.update(logits[:n].detach(), targets)
                total += float(seg.detach())*n
                count += n
                for i, v in enumerate((sep, stab, replay_loss)):
                    sums[i] += float(v.detach())*n
            if distributed:
                for name in ("cm", "bi", "bu"):
                    values = getattr(metric, name).to(device)
                    dist.all_reduce(values)
                    setattr(metric, name, values.cpu())
                values = torch.tensor([total, count, *sums], device=device, dtype=torch.float64)
                dist.all_reduce(values)
                total, count, *sums = values.tolist()
            if rank == 0:
                train_metrics = metric.compute()
                train_metrics["loss"] = total/count
                threshold, threshold_dice = (0.0, float("nan")) if classes != 1 else calibrate_binary_threshold(model, val_loaders[task_id], device)
                val_metrics = evaluate(model, val_loaders[task_id], device, cfg.get("loss"), threshold)
                row = dict(task=task_name, epoch=epoch+1, lr=optimizer.param_groups[0]["lr"], seconds=time.perf_counter()-begin,
                           validation_source="train_diagnostic" if diagnostic_split else "validation")
                for split, values in (("train", train_metrics), ("val", val_metrics)):
                    keys = ("accuracy", "loss", "iou", "dice", "miou", "biou", "foreground_precision", "foreground_recall", "predicted_foreground_fraction", "foreground_collapse")
                    row.update({f"{split}_{k}": values.get(k) for k in keys})
                row.update(separation=sums[0]/count, stability_loss=sums[1]/count, replay_loss=sums[2]/count,
                           decision_threshold=threshold, calibrated_val_dice=threshold_dice)
                history.append(row)
                diagnostics.append(dict(task=task_id, epoch=epoch+1, **controller.diagnostics(model.bottleneck)))
                print(f"{task_name} | epoch [{epoch+1}/{cfg['epochs']}] : train acc {row['train_accuracy']:.4f} / val acc {row['val_accuracy']:.4f} | train loss {row['train_loss']:.4f} / val loss {row['val_loss']:.4f} | train IoU {row['train_iou']:.4f} / val IoU {row['val_iou']:.4f} | train Dice {row['train_dice']:.4f} / val Dice {row['val_dice']:.4f} | mIoU {row['val_miou']:.4f} | BIoU {row['val_biou']:.4f} | val fg P/R/pred {row['val_foreground_precision']:.3f}/{row['val_foreground_recall']:.3f}/{row['val_predicted_foreground_fraction']:.3%} | {row['seconds']:.2f}s", flush=True)
                if row["val_foreground_collapse"]:
                    message = f"Foreground-collapse detected on validation for {task_name}, epoch {epoch+1}: predicted foreground is empty."
                    patience = int(cfg.get("foreground_collapse_patience", 1))
                    if cfg.get("fail_on_foreground_collapse", False) and epoch + 1 >= patience:
                        raise RuntimeError(message)
                    warnings.warn(message)
                score = selection_score(val_metrics)
                if score > best:
                    best = score
                    save_checkpoint(out / f"task_{task_id}_best.pt", dict(model=model.state_dict(), config=cfg, val=val_metrics, epoch=epoch+1, decision_threshold=threshold,
                        validation_source=row["validation_source"],
                        selection="val_dice" if score[0] else "val_loss_fallback_undefined_dice"))
                history_artifacts(out, history, diagnostics)
            scheduler.step()
            if distributed:
                dist.barrier()
        # Carry forward validation-selected weights, not test-selected weights.
        selected = load_checkpoint(out / f"task_{task_id}_best.pt", device)
        model.load_state_dict(selected["model"])
        decision_thresholds[task_id] = selected.get("decision_threshold", 0.0)
        controller.freeze(task_id, model.bottleneck)
        memory_size = cfg.get("replay", {}).get("samples_per_task", 0)
        if memory_size:
            memory = SegmentationDataset(tasks[task_id], manifests[task_id]["splits"]["train"], limit=memory_size)
            replay.extend(memory[i] for i in range(len(memory)))
        if rank == 0:
            diagnostics.append(dict(task=task_id, epoch=cfg["epochs"], boundary=True, **controller.diagnostics(model.bottleneck)))
            history_artifacts(out, history, diagnostics)
            row = [float("nan")]*len(tasks)
            evaluations = {}
            for old_task in range(task_id+1):
                threshold = decision_thresholds.get(old_task, 0.0)
                result = evaluate(model, test_loaders[old_task], device, cfg.get("loss"), threshold)
                result["decision_threshold"] = threshold
                row[old_task] = result["iou"]
                evaluations[tasks[old_task]["name"]] = result
                predictions(model, test_loaders[old_task], out / f"after_{task_id}_task_{old_task}_predictions.png", device, threshold=threshold)
            matrix.append(row)
            write_json(out / f"after_task_{task_id}_test.json", evaluations)
            write_json(out / "test_iou_matrix.json", matrix)
            save_checkpoint(out / f"task_{task_id}_complete.pt", dict(model=model.state_dict(), config=cfg,
                controller=controller.state_dict(), completed_task=task_id, history=history, diagnostics=diagnostics,
                matrix=matrix, replay=replay, prototypes=prototypes, decision_thresholds=decision_thresholds, fingerprints=[m["fingerprint"] for m in manifests]))
        if distributed:
            dist.barrier()
    result = None
    if rank == 0:
        model.eval()
        result = forgetting(matrix)
        result["router"] = router_report(model, prototypes, test_loaders, device)
        result["benchmark_target"] = dict(val_accuracy=.9040, val_dice=.7448, reported_iou=.6277, val_biou=.2740,
            comparable=False, reason="Original benchmark split, task, resolution and aggregation are unspecified")
        result["smoke_only"] = bool(cfg.get("max_samples")) or cfg["model"]["backbone"] != "sam"
        write_json(out / "summary.json", result)
        tsne(model, test_loaders, out, device, cfg["seed"], cfg.get("tsne_samples_per_task", 128))
        print(f"Last-IoU={result['last_iou']:.4f} Avg-IoU={result['avg_iou']:.4f} FF-IoU={result['ff_iou']:.4f}", flush=True)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", help="Trusted task_N_complete.pt; resumes at next task boundary")
    args = parser.parse_args()
    run(args.config, args.resume)


if __name__ == "__main__":
    main()
