import numpy as np
from PIL import Image
import torch

from tqsi.crop_cache import cached_crop
from tqsi.data import SegmentationDataset, read_crop
from tqsi.train import loader


def test_cache_reuses_exact_pixels_and_invalidates_source(tmp_path):
    path = tmp_path / "image.png"
    Image.fromarray(np.arange(192, dtype=np.uint8).reshape(8, 8, 3)).save(path)
    calls = []
    def reader(*args, **kwargs):
        calls.append(1)
        return read_crop(*args, **kwargs)
    args = (reader, path, (1, 2, 7, 8), 4)
    first = cached_crop(*args, rgb=True, directory=tmp_path / "cache")
    second = cached_crop(*args, rgb=True, directory=tmp_path / "cache")
    assert np.array_equal(first, second)
    assert len(calls) == 1
    Image.fromarray(np.zeros((9, 9, 3), dtype=np.uint8)).save(path)
    changed = cached_crop(*args, rgb=True, directory=tmp_path / "cache")
    assert len(calls) == 2
    assert not np.array_equal(first, changed)


def test_cached_dataset_preserves_augmentation_and_targets(tmp_path):
    Image.fromarray(np.arange(192, dtype=np.uint8).reshape(8, 8, 3)).save(tmp_path / "image.png")
    Image.fromarray(np.tri(8, dtype=np.uint8)).save(tmp_path / "mask.png")
    cfg = dict(root=str(tmp_path), image_size=6, tile_size=8,
               foreground_labels=[1], allowed_labels=[0, 1])
    pairs = [dict(id="sample", image="image.png", mask="mask.png")]
    plain = SegmentationDataset(cfg, pairs, augment=True)
    cached = SegmentationDataset(dict(cfg, crop_cache_dir=str(tmp_path / "cache")), pairs, augment=True)
    for seed in range(8):
        torch.manual_seed(seed)
        expected = plain[0]
        torch.manual_seed(seed)
        actual = cached[0]
        assert all(torch.equal(a, b) for a, b in zip(expected, actual))
    assert not list((tmp_path / "cache").rglob("*.tmp"))


def test_prefetch_only_applies_to_workers():
    dataset = torch.utils.data.TensorDataset(torch.arange(4))
    single = loader(dataset, dict(batch_size=2, workers=0, prefetch_factor=4))
    assert len(list(single)) == 2
    workers = loader(dataset, dict(batch_size=2, workers=2, prefetch_factor=4))
    assert workers.prefetch_factor == 4
    assert workers.persistent_workers
