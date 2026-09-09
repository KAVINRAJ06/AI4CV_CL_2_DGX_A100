import numpy as np
from PIL import Image
import yaml

from scripts.prepare_tile_cache import prepare
from tqsi.data import SegmentationDataset, discover


def test_preparation_is_reusable_and_matches_training(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    Image.fromarray(np.arange(8*9*3, dtype=np.uint8).reshape(8, 9, 3)).save(source / "image.png")
    Image.fromarray(np.ones((8, 9), dtype=np.uint8)).save(source / "mask.png")
    task = dict(name="test", root=str(source), image_glob="image.png", mask_glob="mask.png",
                image_suffix="image", mask_suffix="mask", tile_size=4,
                foreground_labels=[1], num_classes=1)
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(task))
    cache = tmp_path / "cache"
    cfg = dict(tasks=[str(path)], image_size=6, crop_cache_dir=str(cache))
    report = prepare(cfg)
    assert report["tasks"][0]["tiles"] == 6
    assert len(list(cache.rglob("*.npy"))) == 12
    def unexpected(*args, **kwargs):
        raise AssertionError("Prepared tiles should not decode/resize again")
    monkeypatch.setattr("scripts.prepare_tile_cache.read_crop", unexpected)
    prepare(cfg)
    pairs, _ = discover(task)
    dataset = SegmentationDataset(dict(task, image_size=6, crop_cache_dir=str(cache)), pairs)
    monkeypatch.setattr("tqsi.data.read_crop", unexpected)
    for image, mask in dataset:
        assert image.shape == (3, 6, 6)
        assert mask.shape == (6, 6)
        assert (mask == 1).all()
