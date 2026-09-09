"""Precompute lossless resized image/mask tiles without loading SAM or a GPU."""
import argparse
import copy
import json
from pathlib import Path
import time

from tqdm import tqdm

from tqsi.config import load_config
from tqsi.crop_cache import cached_crop
from tqsi.data import discover, SegmentationDataset, read_crop


def prepare(config, cache_dir=None):
    cfg = load_config(config) if isinstance(config, (str, Path)) else copy.deepcopy(config)
    directory = cache_dir or cfg.get("crop_cache_dir")
    if not directory:
        raise ValueError("Set crop_cache_dir in the training YAML or pass --cache-dir")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    reports = []
    for task_path in cfg["tasks"]:
        task = load_config(task_path)
        if "dataset_root" in cfg:
            task["root"] = str(Path(cfg["dataset_root"]) / task["relative_root"])
        task["image_size"] = cfg.get("image_size", task.get("image_size", 256))
        pairs, audit = discover(task)
        # All standard tiles, in source order: no random sampling/augmentation,
        # no train/test split changes, and no model or CUDA initialization.
        dataset = SegmentationDataset(task, pairs, augment=False)
        payload_bytes = 0
        for pair, box in tqdm(dataset.samples, desc=f"Prepare {task['name']}", dynamic_ncols=True):
            for kind, rgb in (("image", True), ("mask", False)):
                array = cached_crop(read_crop, dataset.root / pair[kind], box,
                                    dataset.size, rgb=rgb, directory=directory)
                payload_bytes += array.nbytes
        reports.append(dict(task=task["name"], sources=len(pairs), tiles=len(dataset),
                            image_size=dataset.size, payload_bytes=payload_bytes, audit=audit))
    report = dict(cache_dir=str(directory), seconds=time.perf_counter()-started, tasks=reports)
    (directory / "preparation_summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print("Use this same crop_cache_dir in your training YAML. Keep the original dataset_root.", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Your existing training YAML")
    parser.add_argument("--cache-dir", help="Override crop_cache_dir; also set this path in the training YAML")
    args = parser.parse_args()
    prepare(args.config, args.cache_dir)


if __name__ == "__main__":
    main()
