import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn
from torch.nn import functional as F
import yaml

from scripts.prepare_resized_dataset import prepare, resize_for_sam
from tqsi.config import load_config
from tqsi.data import SegmentationDataset, prepare_manifest
from tqsi.model import FrozenSAM, TQSI
from tqsi.prepared import PreparedImages, stack_images, cat_images
from tqsi.train import loader, make_replay_memory, unpack_replay, evaluate


def fail_resize(*args, **kwargs):
    raise AssertionError("Training must not resize prepared input")


def fake_sam_init(self, checkpoint, variant="vit_b"):
    nn.Module.__init__(self)
    self.channels = 8
    encoder = nn.Sequential(nn.AvgPool2d(256), nn.Conv2d(3, 8, 1))
    encoder.img_size = 1024
    self.sam = nn.Module()
    self.sam.image_encoder = encoder
    self.sam.preprocess = lambda x: x/255.
    self.sam.requires_grad_(False)
    self.resize = SimpleNamespace(apply_image_torch=lambda x: F.interpolate(
        x, (1024, 1024), mode="bilinear", align_corners=False, antialias=True))


def make_model(monkeypatch):
    monkeypatch.setattr(FrozenSAM, "__init__", fake_sam_init)
    return TQSI(dict(backbone="sam", sam_checkpoint="unused", bottleneck_type="quantum",
                     n_qubits=3, n_layers=1, decoder_mode="spatial_fpn", decoder_width=8,
                     decoder_dropout=0, adapter_rank=2), 1)


@pytest.fixture
def prepared_data(tmp_path):
    torch.set_num_threads(2)
    root = tmp_path / "original"
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir()
    for index in range(10):
        pixels = np.random.default_rng(index).integers(0, 256, (8, 9, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(root / "images" / f"{index}.png")
        Image.fromarray(np.tri(8, 9, dtype=np.uint8)).save(root / "masks" / f"{index}.png")
    task = dict(name="test", root=str(root), image_glob="images/*.png", mask_glob="masks/*.png",
                tile_size=8, num_classes=1, class_names=["background", "building"],
                foreground_labels=[1], allowed_labels=[0, 1], ignore_labels=[])
    path = tmp_path / "source.yaml"
    path.write_text(yaml.safe_dump(task))
    cfg = dict(tasks=[str(path)], image_size=8, batch_size=2, workers=0, split_seed=42,
               model=dict(backbone="sam", sam_checkpoint="unused"), audit_overlay_count=0)
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*.png")}
    output = prepare(cfg, tmp_path / "resized")
    assert before == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*.png")}
    return cfg, output, root


def test_standalone_data_keeps_splits_and_needs_no_sources(prepared_data, tmp_path, monkeypatch):
    original, output, root = prepared_data
    cfg = load_config(output)
    task = load_config(cfg["tasks"][0])
    manifest = prepare_manifest(task, tmp_path / "run_splits", 42)
    original_manifest = prepare_manifest(load_config(original["tasks"][0]), tmp_path / "original_splits", 42)
    for split in ("train", "val", "test"):
        assert {p["group"] for p in manifest["splits"][split]} == {p["group"] for p in original_manifest["splits"][split]}
    root.rename(root.with_name("original_hidden"))
    monkeypatch.setattr("tqsi.data.read_crop", fail_resize)
    monkeypatch.setattr(Image.Image, "resize", fail_resize)
    dataset = SegmentationDataset(task, manifest["splits"]["train"], augment=True)
    images, targets = next(iter(loader(dataset, cfg, train=True)))
    assert isinstance(images, PreparedImages)
    assert images.image.shape == (2, 3, 8, 8)
    assert images.sam.shape == (2, 3, 1024, 1024)
    assert targets.shape == (2, 8, 8)
    assert all(dataset._foreground_count(i) == pair["foreground_count"]
               for i, (pair, _) in enumerate(dataset.samples))


def test_prepared_augmentation_and_model_gradients(prepared_data, tmp_path, monkeypatch):
    original, output, root = prepared_data
    cfg = load_config(output)
    task = load_config(cfg["tasks"][0])
    manifest = prepare_manifest(task, tmp_path / "splits", 42)
    data = SegmentationDataset(task, manifest["splits"]["train"], augment=True)
    model = make_model(monkeypatch).eval()
    for seed in range(3):
        torch.manual_seed(seed)
        images, target = data[0]
        expected_sam = resize_for_sam((images.image.permute(1, 2, 0)*255).round().byte().numpy())
        np.testing.assert_allclose(images.sam.numpy(), expected_sam, atol=1e-4, rtol=1e-6)
    batch = stack_images([images])
    expected = model(batch.image)
    expected.square().mean().backward()
    gradients = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    model.backbone.resize.apply_image_torch = fail_resize
    actual = model(batch)
    actual.square().mean().backward()
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    for name, parameter in model.named_parameters():
        if name in gradients:
            torch.testing.assert_close(parameter.grad, gradients[name], atol=2e-5, rtol=2e-5)
    assert model.representation(batch).shape[0] == 1


def test_replay_and_evaluation_use_prepared_inputs(prepared_data, tmp_path, monkeypatch):
    _, output, _ = prepared_data
    cfg = load_config(output)
    task = load_config(cfg["tasks"][0])
    manifest = prepare_manifest(task, tmp_path / "splits", 42)
    data = SegmentationDataset(task, manifest["splits"]["train"], limit=2)
    model = make_model(monkeypatch).eval()
    model.backbone.resize.apply_image_torch = fail_resize
    memory = make_replay_memory(model, data, 2, "cpu", 2)
    path = tmp_path / "replay.pt"
    torch.save(memory, path)
    replay, target, teacher = unpack_replay(torch.load(path, weights_only=True), "cpu")
    assert len(replay) == 2 and teacher is not None
    joined = cat_images([replay, replay])
    assert len(joined) == 4
    assert len(joined.split(2)) == 2
    assert model(joined).shape == (4, 1, 8, 8)
    metrics = evaluate(model, loader(data, cfg), "cpu")
    assert np.isfinite(metrics["loss"])


def test_preparation_refuses_existing_or_source_directory(prepared_data):
    cfg, output, root = prepared_data
    with pytest.raises(FileExistsError):
        prepare(cfg, output.parent)
    with pytest.raises(ValueError, match="outside"):
        prepare(cfg, root / "resized")


def test_full_training_run_without_source_or_input_resize(prepared_data, tmp_path, monkeypatch):
    from tqsi.train import run
    _, output, root = prepared_data
    cfg = load_config("configs/base.yaml")
    prepared_cfg = load_config(output)
    cfg.update(prepared_cfg)
    cfg.update(device="cpu", epochs=1, output_dir=str(tmp_path / "training"),
               max_samples=dict(train=2, val=2, test=2), reference_samples=2,
               tsne_samples_per_task=2, train_sampling={"enabled": False},
               replay=dict(samples_per_task=2, batch_size=1), timing=False)
    cfg["model"].update(bottleneck_type="quantum", n_qubits=3, n_layers=1,
                       decoder_mode="spatial_fpn", decoder_width=8, decoder_dropout=0)
    def prepared_only_init(self, checkpoint, variant="vit_b"):
        fake_sam_init(self, checkpoint, variant)
        self.resize.apply_image_torch = fail_resize
    monkeypatch.setattr(FrozenSAM, "__init__", prepared_only_init)
    root.rename(root.with_name("source_not_available"))
    monkeypatch.setattr("tqsi.data.read_crop", fail_resize)
    result = run(cfg)
    assert np.isfinite(result["last_iou"])
    assert (tmp_path / "training" / "task_0_complete.pt").is_file()
    assert (tmp_path / "training" / "after_0_task_0_predictions.png").is_file()
