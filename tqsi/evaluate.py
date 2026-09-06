"""Evaluate a trusted trained checkpoint against its persisted test split."""
import argparse
import json
from pathlib import Path
import torch
from .artifacts import write_json, predictions
from .data import SegmentationDataset, prepare_manifest
from .model import TQSI
from .train import load_checkpoint, loader, evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dataset-root", help="Relocate the same datasets to another machine")
    parser.add_argument("--sam-checkpoint", help="Relocate the original pretrained SAM checkpoint")
    args = parser.parse_args()
    path = Path(args.checkpoint)
    state = load_checkpoint(path)
    cfg = state["config"]
    if args.sam_checkpoint:
        cfg["model"]["sam_checkpoint"] = args.sam_checkpoint
    tasks = json.loads((path.parent / "datasets.json").read_text())
    model = TQSI(cfg["model"], len(tasks)).to(args.device)
    model.load_state_dict(state["model"])
    model.eval()
    report = {}
    for task in tasks:
        if args.dataset_root:
            task["root"] = str(Path(args.dataset_root) / task["relative_root"])
        manifest = prepare_manifest(task, path.parent / "splits", cfg.get("split_seed", 42))
        batches = loader(SegmentationDataset(task, manifest["splits"]["test"], limit=cfg.get("max_samples", {}).get("test")), cfg)
        report[task["name"]] = evaluate(model, batches, args.device)
        predictions(model, batches, path.parent / f"eval_{task['name']}.png", args.device)
    write_json(path.parent / "reevaluation.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
