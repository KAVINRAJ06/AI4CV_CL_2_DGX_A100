import torch
from tqsi.bottlenecks import QuantumBottleneck
from tqsi.continual import TaskController


def test_unneeded_auxiliary_losses_do_not_simulate(monkeypatch):
    model = QuantumBottleneck(n_qubits=3, n_layers=2, n_tasks=2)
    controller = TaskController()
    controller.register(0, torch.randn(2, 8))
    def unexpected(_):
        raise AssertionError("Unneeded state simulation")
    monkeypatch.setattr(model, "get_state", unexpected)
    assert all(v.item() == 0 for v in controller.losses(model))
    controller.register(1, torch.randn(2, 8))
    controller.frozen[0] = torch.randn(2, 8)
    assert all(v.item() == 0 for v in controller.losses(model, False, False))


def test_stability_retains_gradient_and_only_simulates_old_tasks(monkeypatch):
    model = QuantumBottleneck(n_qubits=3, n_layers=2, n_tasks=2)
    controller = TaskController()
    controller.register(0, torch.randn(2, 8))
    controller.freeze(0, model)
    controller.register(1, torch.randn(2, 8))
    with torch.no_grad():
        model.weights.add_(.2)
    original = model.get_state
    calls = []
    def tracked(x):
        calls.append(x)
        return original(x)
    monkeypatch.setattr(model, "get_state", tracked)
    sep, stab = controller.losses(model, False, True)
    assert len(calls) == 1 and sep.item() == 0 and stab.item() > 0
    stab.backward()
    assert model.weights.grad.abs().sum() > 1e-6
