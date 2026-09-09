"""Standalone pre-resized datasets and paired decoder/SAM image batches."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate


@dataclass
class PreparedImages:
    image: torch.Tensor  # Original decoder resolution, float RGB [0, 1].
    sam: torch.Tensor    # Pre-resized encoder resolution, float RGB [0, 255].

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        return PreparedImages(self.image[index], self.sam[index])

    def to(self, *args, **kwargs):
        return PreparedImages(self.image.to(*args, **kwargs), self.sam.to(*args, **kwargs))

    def pin_memory(self):
        return PreparedImages(self.image.pin_memory(), self.sam.pin_memory())

    def split(self, size):
        return [PreparedImages(a, b) for a, b in zip(self.image.split(size), self.sam.split(size))]


def stack_images(images):
    if isinstance(images[0], PreparedImages):
        return PreparedImages(torch.stack([x.image for x in images]), torch.stack([x.sam for x in images]))
    return torch.stack(images)


def cat_images(images):
    if isinstance(images[0], PreparedImages):
        if not all(isinstance(x, PreparedImages) for x in images):
            raise ValueError("Prepared training requires prepared replay; start a fresh run")
        return PreparedImages(torch.cat([x.image for x in images]), torch.cat([x.sam for x in images]))
    return torch.cat(images)


def collate_images(batch):
    images, targets = zip(*batch)
    return stack_images(images), default_collate(targets)


def load_sample(root, pair, size, augment):
    image = np.load(root / pair["image"], allow_pickle=False)
    sam = np.load(root / pair["sam_image"], allow_pickle=False)
    target = np.load(root / pair["mask"], allow_pickle=False)
    if image.shape != (size, size, 3) or target.shape != (size, size) or sam.shape != (3, 1024, 1024):
        raise ValueError(f"Prepared sample geometry differs: {pair['id']}; regenerate the dataset")
    if image.dtype != np.uint8 or sam.dtype != np.float32 or target.dtype != np.int64:
        raise ValueError(f"Prepared sample dtype differs: {pair['id']}")
    if augment:
        if torch.rand(()) < .5:
            image, target, sam = np.flip(image, 1), np.flip(target, 1), np.flip(sam, 2)
        if torch.rand(()) < .5:
            image, target, sam = np.flip(image, 0), np.flip(target, 0), np.flip(sam, 1)
        turns = int(torch.randint(4, ()))
        image, target = np.rot90(image, turns), np.rot90(target, turns)
        sam = np.rot90(sam, turns, axes=(1, 2))
    return (PreparedImages(torch.from_numpy(image.copy()).permute(2, 0, 1).float()/255.,
                           torch.from_numpy(sam.copy())), torch.from_numpy(target.copy()))


def prepared_manifest(cfg, directory, seed):
    root = Path(cfg["root"])
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["seed"] != seed or manifest["image_size"] != cfg["image_size"]:
        raise ValueError("Prepared dataset split seed/image_size differs; regenerate it")
    if manifest["name"] != cfg["name"] or manifest["schema"] != "prepared_sam_v1":
        raise ValueError("Prepared dataset identity/schema differs")
    for key in ("num_classes", "class_names", "foreground_labels", "allowed_labels", "ignore_labels", "color_map", "label_map"):
        if manifest["label_config"].get(key) != json.loads(json.dumps(cfg.get(key))):
            raise ValueError(f"Prepared dataset label configuration differs at {key}")
    destination = Path(directory) / f"{cfg['name']}.json"
    if destination.exists():
        previous = json.loads(destination.read_text())
        if previous["fingerprint"] != manifest["fingerprint"]:
            raise ValueError("Prepared dataset differs from existing run; use a fresh output_dir")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(manifest, indent=2))
    return manifest
