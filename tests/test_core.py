import numpy as np
import pytest
import torch
from tqsi.bottlenecks import QuantumBottleneck, OrthogonalBottleneck, MLPBottleneck, normalize
from tqsi.continual import fidelity, masked_step, segmentation_loss
from tqsi.metrics import Metrics, forgetting
from tqsi.data import split_pairs, SegmentationDataset, discover, prepare_manifest
from tqsi.model import TQSI


@pytest.mark.parametrize("factory", [QuantumBottleneck, OrthogonalBottleneck, MLPBottleneck])
def test_shapes_gradients_and_isolation(factory):
    model = factory(n_qubits=3, n_layers=2, n_tasks=2, hidden_dim=8)
    x = normalize(torch.randn(3, 8)).requires_grad_()
    out = model(x)
    assert out.shape == (3, 9)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.03, weight_decay=.1)
    for task in [0, 0, 1, 1]:
        masked = model.masked_parameters(task)
        before = [p.detach().clone() for p, _ in masked]
        loss = model(x).square().mean()
        masked_step(loss, optimizer, masked, list(model.parameters()))
        assert x.grad is not None and torch.isfinite(x.grad).all()
        assert x.grad.abs().sum() > 0
        for (p, mask), old in zip(masked, before):
            assert torch.equal(p[~mask], old[~mask])


def test_quantum_readout_matches_pennylane():
    import pennylane as qml
    model = QuantumBottleneck(n_qubits=3, n_layers=2, n_tasks=2)
    x = normalize(torch.randn(2, 8))
    @qml.qnode(qml.device("default.qubit", wires=3), interface="torch")
    def expected(x):
        qml.AmplitudeEmbedding(x, wires=range(3))
        qml.StronglyEntanglingLayers(model.weights, wires=range(3), ranges=[1, 1])
        return [qml.expval(op(w)) for w in range(3) for op in (qml.PauliX, qml.PauliY, qml.PauliZ)]
    assert torch.allclose(model(x), torch.stack(expected(x), -1).float(), atol=1e-6)
    assert torch.allclose(model.get_state(x).abs().square().sum(-1), torch.ones(2), atol=1e-5)


@pytest.mark.parametrize("factory", [QuantumBottleneck, OrthogonalBottleneck])
def test_common_unitary_separation_is_invariant(factory):
    model = factory(n_qubits=3, n_layers=2, n_tasks=2)
    x = normalize(torch.randn(2, 8))
    state = model.get_state(x)
    overlap = fidelity(state[0], state[1])
    assert torch.allclose(overlap, fidelity(x[0], x[1]), atol=2e-6)
    parameter = model.weights if hasattr(model, "weights") else model.A
    grad, = torch.autograd.grad(overlap, parameter)
    assert grad.abs().max() < 2e-6


def test_orthogonal_and_zero_amplitudes():
    model = OrthogonalBottleneck(n_qubits=3)
    w = model.matrix()
    assert torch.allclose(w @ w.T, torch.eye(8), atol=1e-6)
    assert torch.equal(normalize(torch.zeros(8)), torch.tensor([1., 0, 0, 0, 0, 0, 0, 0]))
    with pytest.raises(ValueError):
        QuantumBottleneck(n_qubits=2, n_tasks=3)


def test_known_metrics():
    gt = torch.tensor([[[0, 1], [1, 1]]])
    pred = torch.tensor([[[[-9., 9.], [-9., 9.]]]])
    metric = Metrics()
    metric.update(pred, gt)
    result = metric.compute()
    assert result["accuracy"] == .75
    assert result["iou"] == pytest.approx(2/3)
    assert result["dice"] == pytest.approx(4/5)
    assert result["miou"] == pytest.approx((2/3+1/2)/2)
    perfect = Metrics()
    perfect.update(gt[:, None]*20.-10., gt)
    assert perfect.compute()["biou"] == 1
    assert forgetting([[.8, np.nan], [.6, .9]])["ff_iou"] == pytest.approx(.2)
    assert forgetting([[.8]])["ff_iou"] == 0


def test_multiclass_and_ignore():
    logits = torch.randn(2, 3, 8, 8, requires_grad=True)
    target = torch.randint(3, (2, 8, 8))
    target[:, :2] = -100
    loss = segmentation_loss(logits, target)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert (logits.grad[:, :, :2] == 0).all()
    metric = Metrics(3)
    metric.update(logits, target)
    assert metric.cm.sum() == 96


def test_source_group_splits():
    pairs = [dict(id=f"{i}_{j}", group=str(i)) for i in range(20) for j in range(3)]
    splits = split_pairs(pairs)
    assert [len(s) for s in splits.values()] == [42, 12, 6]
    groups = [{p["group"] for p in s} for s in splits.values()]
    assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
    assert splits == split_pairs(pairs)


def test_folder_adapter(tmp_path):
    from PIL import Image
    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()
    for i in range(10):
        Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(tmp_path / f"images/{i}.png")
        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[2:8] = 255
        Image.fromarray(mask).save(tmp_path / f"masks/{i}_mask.png")
    cfg = dict(name="fake", root=str(tmp_path), image_glob="images/*.png", mask_glob="masks/*.png", mask_suffix="_mask", foreground_labels=[255], allowed_labels=[0, 255], image_size=8, tile_size=8)
    pairs, _ = discover(cfg)
    manifest = prepare_manifest(cfg, tmp_path / "splits")
    dataset = SegmentationDataset(cfg, manifest["splits"]["train"])
    assert len(dataset) == 28
    image, mask = dataset[0]
    assert image.shape == (3, 8, 8) and mask.shape == (8, 8)
    assert set(mask.unique().tolist()) == {0, 1}
    cfg["group_regex"] = r"^(\d)"
    with pytest.raises(ValueError, match="changed"):
        prepare_manifest(cfg, tmp_path / "splits")


def test_tensor_model_frozen_backbone_gradient():
    model = TQSI(dict(backbone="tiny", bottleneck_type="quantum", n_qubits=3, n_layers=2), 2)
    images = torch.rand(2, 3, 16, 16)
    logits = model(images)
    assert logits.shape == (2, 1, 16, 16)
    segmentation_loss(logits, torch.randint(2, (2, 16, 16))).backward()
    assert all(p.grad is None and not p.requires_grad for p in model.backbone.parameters())
    assert model.decoder_head.weight.grad.abs().sum() > 0
    assert model.projection[2].weight.grad.abs().sum() > 0


def test_empty_foreground_checkpoint_selection():
    from tqsi.train import selection_score
    assert selection_score(dict(dice=float('nan'), loss=1.)) > (-1, -float('inf'))
    assert selection_score(dict(dice=0., loss=2.)) > selection_score(dict(dice=float('nan'), loss=1.))


def test_tiff_crop_matches_pillow(tmp_path):
    from PIL import Image
    from tqsi.data import read_crop
    image = np.random.default_rng(7).integers(0, 256, (40, 50, 3), dtype=np.uint8)
    path = tmp_path / 'source.tif'
    Image.fromarray(image).save(path)
    expected = np.array(Image.fromarray(image).crop((3, 5, 20, 25)).resize((16, 16), Image.Resampling.BILINEAR))
    assert np.array_equal(read_crop(path, (3, 5, 20, 25), 16, rgb=True), expected)


def test_invalid_backbone_never_falls_back_to_tiny():
    with pytest.raises(ValueError, match='explicitly'):
        TQSI(dict(backbone='sma', bottleneck_type='quantum'), 2)
