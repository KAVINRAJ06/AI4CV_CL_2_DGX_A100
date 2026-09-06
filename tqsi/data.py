"""Dataset adapters end at the (RGB float tensor, indexed mask tensor) boundary."""
from pathlib import Path
import hashlib
import json
import random
import re
from functools import lru_cache
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, Sampler, WeightedRandomSampler


class DistributedWeightedSampler(Sampler):
    """Deterministic weighted replacement sampling with equal work on each DDP rank."""
    def __init__(self, weights, replicas, rank, seed):
        self.weights, self.replicas, self.rank, self.seed, self.epoch = weights, replicas, rank, seed, 0
        self.samples_per_rank = (len(weights) + replicas - 1) // replicas

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        selected = torch.multinomial(self.weights, self.samples_per_rank*self.replicas, replacement=True, generator=generator)
        return iter(selected[self.rank::self.replicas].tolist())

    def __len__(self):
        return self.samples_per_rank


@lru_cache(maxsize=8)
def mapped_tiff(path):
    """Per-worker bounded memory maps avoid decoding whole large TIFFs per tile."""
    import tifffile
    try:
        data = tifffile.memmap(path, mode="r")
        if data.ndim in (2, 3) and (data.ndim == 2 or data.shape[-1] in (1, 3, 4)):
            return data
    except (ValueError, OSError):
        pass  # Compressed/non-contiguous TIFFs use the Pillow fallback.
    return None


def read_crop(path, box, size, rgb=False):
    image = read_native_crop(path, box)
    if rgb:
        image = image.convert("RGB")
    return np.array(image.resize((size, size), Image.Resampling.BILINEAR if rgb else Image.Resampling.NEAREST))


def read_native_crop(path, box):
    data = mapped_tiff(str(path)) if path.suffix.lower() in (".tif", ".tiff") else None
    if data is not None:
        left, top, right, bottom = box
        image = Image.fromarray(np.array(data[top:bottom, left:right]))
    else:
        with Image.open(path) as source:
            image = source.crop(box)
    return image


def discover(cfg):
    root = Path(cfg["root"]).expanduser().resolve()
    def index(pattern, suffix):
        result = {}
        for path in sorted(root.glob(pattern)):
            if not path.is_file():
                continue
            key = path.stem
            if suffix:
                if not key.endswith(suffix):
                    raise ValueError(f"Missing configured suffix {suffix}: {path}")
                key = key[:-len(suffix)]
            if key in result:
                raise ValueError(f"Duplicate sample ID {key}; use globally unique stems")
            result[key] = path.relative_to(root).as_posix()
        return result
    images = index(cfg["image_glob"], cfg.get("image_suffix", ""))
    masks = index(cfg["mask_glob"], cfg.get("mask_suffix", ""))
    if not images or not masks:
        raise ValueError(f"No images/masks found under {root}")
    missing = sorted(images.keys() - masks.keys())
    orphan = sorted(masks.keys() - images.keys())
    if (missing or orphan) and not cfg.get("allow_unpaired", False):
        raise ValueError(f"Unpaired samples: {len(missing)} images, {len(orphan)} masks. Examples: {missing[:3]}, {orphan[:3]}")
    pairs = []
    for key in sorted(images.keys() & masks.keys()):
        group = key
        if cfg.get("group_regex"):
            match = re.search(cfg["group_regex"], key)
            if not match:
                raise ValueError(f"Group regex does not match {key}")
            group = match.group(1)
        pairs.append(dict(id=key, image=images[key], mask=masks[key], group=group))
    return pairs, dict(unpaired_images=missing, unpaired_masks=orphan)


def split_pairs(pairs, seed=42):
    """Split source groups BEFORE tiling. Integer counts use largest remainder."""
    groups = sorted({p["group"] for p in pairs})
    if len(groups) < 10:
        raise ValueError("At least 10 independent source groups are required for 70/20/10 splitting")
    random.Random(seed).shuffle(groups)
    expected = np.array([.7, .2, .1]) * len(groups)
    counts = np.floor(expected).astype(int)
    for i in np.argsort(-(expected - counts), kind="stable")[:len(groups) - counts.sum()]:
        counts[i] += 1
    a, b = counts[0], counts[0] + counts[1]
    lookup = {g: s for s, gs in zip(("train", "val", "test"), (groups[:a], groups[a:b], groups[b:])) for g in gs}
    return {s: [p for p in pairs if lookup[p["group"]] == s] for s in ("train", "val", "test")}


def prepare_manifest(cfg, directory, seed=42):
    pairs, audit = discover(cfg)
    root = Path(cfg["root"])
    signature = dict(pairs=pairs, seed=seed, schema=1, group_regex=cfg.get("group_regex"),
                     files=[(p[k], (root / p[k]).stat().st_size, (root / p[k]).stat().st_mtime_ns) for p in pairs for k in ("image", "mask")])
    fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    path = Path(directory) / f"{cfg['name']}.json"
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest["fingerprint"] != fingerprint:
            raise ValueError(f"Dataset/split changed: {path}. Choose a new run directory; do not silently reuse splits.")
        return manifest
    manifest = dict(fingerprint=fingerprint, seed=seed, ratios=[.7, .2, .1], audit=audit, splits=split_pairs(pairs, seed))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))
    return manifest


class SegmentationDataset(Dataset):
    def __init__(self, cfg, pairs, augment=False, limit=None, sampling=None):
        self.cfg, self.augment = cfg, augment
        self.root = Path(cfg["root"])
        self.size = int(cfg.get("image_size", 256))
        tile = int(cfg.get("tile_size", 512))
        self.samples = []
        self.sampling = sampling or {}
        self._foreground_counts = {}
        for pair in pairs:
            with Image.open(self.root / pair["image"]) as im:
                width, height = im.size
            with Image.open(self.root / pair["mask"]) as mask:
                if mask.size != (width, height):
                    raise ValueError(f"Image/mask size mismatch: {pair['id']}")
            if tile:
                for top in range(0, height, tile):
                    for left in range(0, width, tile):
                        self.samples.append((pair, (left, top, min(left+tile, width), min(top+tile, height))))
            else:
                self.samples.append((pair, (0, 0, width, height)))
        if limit and len(self.samples) > limit:
            self._limit_samples(limit)

    def __len__(self):
        return len(self.samples)

    def _target_from_raw_mask(self, mask, pair):
        if mask.ndim == 3:
            colors = self.cfg.get("color_map")
            if not colors:
                raise ValueError("RGB masks require color_map: {'r,g,b': integer_label}")
            converted = np.full(mask.shape[:2], -999, dtype=np.int64)
            for color, label in colors.items():
                converted[(mask[..., :3] == [int(x) for x in color.split(",")]).all(-1)] = label
            if (converted == -999).any():
                raise ValueError(f"Unmapped RGB mask colors in {pair['mask']}")
            mask = converted
        ignored = np.isin(mask, self.cfg.get("ignore_labels", []))
        allowed = self.cfg.get("allowed_labels")
        if allowed is not None and not np.isin(mask, allowed + self.cfg.get("ignore_labels", [])).all():
            raise ValueError(f"Unknown labels {np.unique(mask)} in {pair['mask']}")
        if self.cfg.get("num_classes", 1) == 1:
            mask = np.isin(mask, self.cfg["foreground_labels"]).astype(np.int64)
        else:
            mapped = np.full(mask.shape, -100, dtype=np.int64)
            for raw, target in self.cfg["label_map"].items():
                mapped[mask == int(raw)] = int(target)
            if np.any((mapped == -100) & ~ignored):
                raise ValueError(f"Unmapped labels in {pair['mask']}")
            mask = mapped
            if np.any((mask >= self.cfg["num_classes"]) | ((mask < 0) & (mask != -100))):
                raise ValueError("label_map targets must lie in [0, num_classes)")
        mask[ignored] = -100
        return mask

    def _foreground_count(self, index):
        """Count native-resolution foreground pixels once per tile for sampling."""
        if index not in self._foreground_counts:
            pair, box = self.samples[index]
            target = self._target_from_raw_mask(np.array(read_native_crop(self.root / pair["mask"], box)), pair)
            if self.cfg.get("num_classes", 1) == 1:
                self._foreground_counts[index] = int((target == 1).sum())
            else:
                self._foreground_counts[index] = int((target > 0).sum())
        return self._foreground_counts[index]

    def _limit_samples(self, limit):
        """Bounded local runs retain the requested mix of positive/background tiles."""
        if not self.sampling.get("enabled", False):
            ids = np.random.default_rng(123).choice(len(self.samples), limit, replace=False)
        else:
            minimum = int(self.sampling.get("min_foreground_pixels", 16))
            # A bounded smoke run must not scan every high-resolution tile just
            # to select a small subset. Full DGX runs (no `limit`) sample all.
            pool_size = min(len(self.samples), int(self.sampling.get("candidate_pool_size", max(512, limit * 10))))
            candidates = np.random.default_rng(123).choice(len(self.samples), pool_size, replace=False).tolist()
            positive = [i for i in candidates if self._foreground_count(i) >= minimum]
            positive_set = set(positive)
            negative = [i for i in candidates if i not in positive_set]
            target = int(round(limit * float(self.sampling.get("target_foreground_fraction", .5))))
            target = min(target, len(positive))
            rng = np.random.default_rng(123)
            picked = rng.choice(positive, target, replace=False).tolist() if target else []
            remaining = limit - len(picked)
            pool = negative if len(negative) >= remaining else [i for i in candidates if i not in picked]
            if len(pool) < remaining:
                raise ValueError("Sampling candidate pool is smaller than the requested bounded dataset")
            picked += rng.choice(pool, remaining, replace=False).tolist()
            ids = np.array(sorted(picked))
        old = self.samples
        patch = int(self.sampling.get("foreground_crop_size", 0))
        self.samples = [
            self._foreground_crop(old[i], patch) if patch and self._foreground_count(i) >= minimum else old[i]
            for i in ids
        ]
        self._foreground_counts = {}

    def _foreground_crop(self, sample, crop_size):
        """Replace a positive source tile with a deterministic foreground-centred patch."""
        pair, box = sample
        left, top, right, bottom = box
        width, height = right-left, bottom-top
        crop_w, crop_h = min(crop_size, width), min(crop_size, height)
        target = self._target_from_raw_mask(np.array(read_native_crop(self.root / pair["mask"], box)), pair)
        positive = np.argwhere(target == 1 if self.cfg.get("num_classes", 1) == 1 else target > 0)
        if not len(positive):
            return sample
        y, x = np.median(positive, axis=0).astype(int)
        new_left = int(np.clip(left + x - crop_w//2, left, right-crop_w))
        new_top = int(np.clip(top + y - crop_h//2, top, bottom-crop_h))
        return pair, (new_left, new_top, new_left+crop_w, new_top+crop_h)

    def training_sampler(self, seed, replicas=1, rank=0):
        if not self.sampling.get("enabled", False):
            return None
        minimum = int(self.sampling.get("min_foreground_pixels", 16))
        target = float(self.sampling.get("target_foreground_fraction", .5))
        if not 0 < target < 1:
            raise ValueError("target_foreground_fraction must be strictly between 0 and 1")
        positive = torch.tensor([self._foreground_count(i) >= minimum for i in range(len(self))])
        n_pos, n_neg = int(positive.sum()), len(self)-int(positive.sum())
        if not n_pos:
            raise ValueError(f"No foreground tiles meet min_foreground_pixels={minimum}; verify labels or lower the threshold")
        if not n_neg:
            return None
        weights = torch.where(positive, torch.tensor(target/n_pos), torch.tensor((1-target)/n_neg)).double()
        if replicas == 1:
            return WeightedRandomSampler(weights, num_samples=len(self), replacement=True,
                                         generator=torch.Generator().manual_seed(seed))
        return DistributedWeightedSampler(weights, replicas, rank, seed)

    def __getitem__(self, index):
        pair, box = self.samples[index]
        image = read_crop(self.root / pair["image"], box, self.size, rgb=True)
        mask = self._target_from_raw_mask(read_crop(self.root / pair["mask"], box, self.size), pair)
        if self.augment:
            if torch.rand(()) < .5:
                image, mask = np.flip(image, 1), np.flip(mask, 1)
            if torch.rand(()) < .5:
                image, mask = np.flip(image, 0), np.flip(mask, 0)
            turns = int(torch.randint(4, ()))
            image, mask = np.rot90(image, turns), np.rot90(mask, turns)
        return torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255., torch.from_numpy(mask.copy()).long()
