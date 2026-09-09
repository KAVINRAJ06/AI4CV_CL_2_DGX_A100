"""Create a separate, standalone dataset with all input resizing done offline."""
import argparse
import copy
import hashlib
import json
from itertools import groupby
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from tqdm import tqdm
import yaml

from tqsi.config import load_config
from tqsi.data import prepare_manifest, SegmentationDataset


def resize_for_sam(image):
    # Match the existing float RGB -> SAM interpolation, avoiding uint8 rounding
    # or float16 quantization of the encoder's prepared input.
    tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float()/255.
    return F.interpolate((tensor*255.)[None], (1024, 1024), mode="bilinear",
                         align_corners=False, antialias=True)[0].numpy()


def prepare(config, output):
    cfg = load_config(config) if isinstance(config, (str, Path)) else copy.deepcopy(config)
    if cfg["model"]["backbone"] != "sam":
        raise ValueError("This prepared dataset is for the 1024px SAM backbone")
    if cfg.get("train_sampling", {}).get("foreground_crop_size") or cfg.get("bounded_evaluation_sampling", {}).get("foreground_crop_size"):
        raise ValueError("Remove foreground_crop_size for this workflow; offline datasets preserve standard source tiles")
    output = Path(output).resolve()
    tasks = []
    for path in cfg["tasks"]:
        task = load_config(path)
        if task.get("prepared_sam"):
            raise ValueError("Use the original training YAML as the preparation input")
        if "dataset_root" in cfg:
            task["root"] = str(Path(cfg["dataset_root"]) / task["relative_root"])
        root = Path(task["root"]).resolve()
        if output == root or root in output.parents:
            raise ValueError("Output must be outside every original dataset directory")
        task["root"] = str(root)
        task["image_size"] = int(cfg.get("image_size", task.get("image_size", 256)))
        if not task["name"] or Path(task["name"]).name != task["name"] or task["name"] in (".", ".."):
            raise ValueError("Task names must be simple directory names")
        tasks.append(task)
    if len({t["name"] for t in tasks}) != len(tasks):
        raise ValueError("Task names must be unique")
    # Never overwrite source data or a previously prepared dataset.
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    configs = []
    for task in tasks:
        folder = output / task["name"]
        folder.mkdir()
        manifest = prepare_manifest(task, output / "source_splits", cfg.get("split_seed", 42))
        audit = dict(task=task["name"], raw_label_histogram={}, splits={}, alignment_errors=[])
        result = dict(schema="prepared_sam_v1", name=task["name"], image_size=task["image_size"],
                      seed=manifest["seed"], source_fingerprint=manifest["fingerprint"],
                      audit=manifest["audit"], preparation_audit=audit, splits={})
        result["label_config"] = {key: task.get(key) for key in
            ("num_classes", "class_names", "foreground_labels", "allowed_labels", "ignore_labels", "color_map", "label_map")}
        for split, pairs in manifest["splits"].items():
            dataset = SegmentationDataset(task, pairs, augment=False)
            split_dir = folder / split
            split_dir.mkdir()
            estimated = len(dataset) * (3*1024*1024*4 + task["image_size"]**2*11)
            print(f"{task['name']}/{split}: {len(dataset)} tiles, approximately {estimated/1024**3:.2f} GiB", flush=True)
            entries = []
            foreground_total = foreground_tiles = area = 0
            # Decode each original source once, then release it after its tiles.
            # Training will only open the resulting arrays, never these sources.
            with tqdm(total=len(dataset), desc=f"Resize {task['name']}/{split}", dynamic_ncols=True) as progress:
                for _, samples in groupby(dataset.samples, key=lambda item: item[0]["id"]):
                    samples = list(samples)
                    pair = samples[0][0]
                    with Image.open(dataset.root / pair["image"]) as source_image, Image.open(dataset.root / pair["mask"]) as source_mask:
                        source_image.load()
                        source_mask.load()
                        raw = np.asarray(source_mask)
                        if raw.ndim == 2:
                            labels, counts = np.unique(raw, return_counts=True)
                            for label, count in zip(labels, counts):
                                key = str(int(label))
                                audit["raw_label_histogram"][key] = audit["raw_label_histogram"].get(key, 0) + int(count)
                        del raw
                        for pair, box in samples:
                            image = np.array(source_image.crop(box).convert("RGB").resize((dataset.size, dataset.size), Image.Resampling.BILINEAR))
                            native_mask = source_mask.crop(box)
                            native_target = dataset._target_from_raw_mask(np.array(native_mask), pair)
                            foreground = int((native_target == 1 if task.get("num_classes", 1) == 1 else native_target > 0).sum())
                            target = dataset._target_from_raw_mask(np.array(native_mask.resize((dataset.size, dataset.size), Image.Resampling.NEAREST)), pair)
                            foreground_total += foreground
                            foreground_tiles += foreground > 0
                            area += native_target.size
                            index = len(entries)
                            prefix = f"{split}/{index:08d}"
                            files = dict(image=f"{prefix}_image.npy", sam_image=f"{prefix}_sam1024.npy", mask=f"{prefix}_mask.npy")
                            np.save(folder / files["image"], image, allow_pickle=False)
                            np.save(folder / files["sam_image"], resize_for_sam(image), allow_pickle=False)
                            np.save(folder / files["mask"], target, allow_pickle=False)
                            entries.append(dict(id=f"{pair['id']}:{index}", source_id=pair["id"], group=pair["group"],
                                                source_box=list(box), foreground_count=foreground, **files))
                            progress.update(1)
            audit["splits"][split] = dict(sources=len(pairs), tiles=len(dataset), foreground_tiles=int(foreground_tiles),
                                          foreground_fraction=foreground_total/max(1, area))
            result["splits"][split] = entries
        if any(not audit["splits"][split]["foreground_tiles"] for split in ("train", "val")):
            raise ValueError("Train and validation must each contain foreground; verify the source label configuration")
        result["fingerprint"] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
        (folder / "manifest.json").write_text(json.dumps(result, indent=2))
        prepared_task = dict(task, root=str(folder), relative_root=task["name"], prepared_sam=True)
        prepared_task.pop("crop_cache_dir", None)
        task_path = folder / "dataset.yaml"
        task_path.write_text(yaml.safe_dump(prepared_task, sort_keys=False))
        configs.append(str(task_path))
    # Fully resolved copy: all original training settings remain unless they
    # select input data, logging overhead, or the new output directory.
    cfg.pop("crop_cache_dir", None)
    cfg["dataset_root"] = str(output)
    cfg["tasks"] = configs
    cfg["output_dir"] = str(output / "training_run")
    cfg["timing"] = False
    cfg["model"]["sam_checkpoint"] = str(Path(cfg["model"]["sam_checkpoint"]).resolve())
    cfg["prepared_dataset"] = True
    destination = output / "training.yaml"
    destination.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"Prepared dataset complete in {time.perf_counter()-started:.1f}s. Original data unchanged.", flush=True)
    print(f'Run: python -u -m tqsi.train --config "{destination}"', flush=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Original training YAML")
    parser.add_argument("--output", required=True, help="New separate dataset directory; must not exist")
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    prepare(args.config, args.output)


if __name__ == "__main__":
    main()
