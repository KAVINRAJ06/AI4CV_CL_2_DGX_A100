"""Shared-current-weight TQSI circuit and B1/B2 controls from the specification."""
import torch
from torch import nn
from torch.nn import functional as F


def normalize(x):
    norm = x.norm(dim=-1, keepdim=True)
    fallback = torch.zeros_like(x)
    fallback[..., 0] = 1
    return torch.where(norm > 1e-8, x / norm.clamp_min(1e-8), fallback)


def partition(shape, axis, tasks):
    if tasks < 1 or tasks > shape[axis]:
        raise ValueError(f"Cannot allocate {tasks} nonempty tasks along axis of size {shape[axis]}")
    masks = torch.zeros((tasks, *shape), dtype=torch.bool)
    for task, ids in enumerate(torch.tensor_split(torch.arange(shape[axis]), tasks)):
        region = [slice(None)] * len(shape)
        region[axis] = ids
        masks[(task, *region)] = True
    return masks


class QuantumBottleneck(nn.Module):
    def __init__(self, n_qubits=8, n_layers=6, n_tasks=2, mask_strategy="qubit", **_):
        super().__init__()
        import pennylane as qml
        self.feature_dim, self.readout_dim = 2**n_qubits, 3*n_qubits
        self.weights = nn.Parameter(.01 * torch.randn(n_layers, n_qubits, 3))
        if mask_strategy not in ("layer", "qubit"):
            raise ValueError("mask_strategy must be layer or qubit")
        self.register_buffer("masks", partition(self.weights.shape, int(mask_strategy == "qubit"), n_tasks))
        # default.qubit supports input gradients, complex states and Torch CUDA tensors.
        # lightning.gpu adjoint cannot simply replace this differentiable state QNode.
        device = qml.device("default.qubit", wires=n_qubits, shots=None)
        @qml.qnode(device, interface="torch", diff_method="backprop")
        def state(x, weights):
            qml.AmplitudeEmbedding(x, wires=range(n_qubits), normalize=False)
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits), ranges=[1]*n_layers)
            return qml.state()
        self._state = state
        indices = torch.arange(self.feature_dim)
        flips, signs = [], []
        for wire in range(n_qubits):
            bit = 1 << (n_qubits-wire-1)
            flips.append(indices ^ bit)
            signs.append(1 - 2*((indices & bit) != 0).long())
        self.register_buffer("flips", torch.stack(flips))
        self.register_buffer("signs", torch.stack(signs))

    def get_state(self, x):
        # One broadcast QNode, not one Python QNode invocation per sample.
        state = self._state(normalize(x).float(), self.weights.float())
        return state.to(device=x.device, dtype=torch.complex64)

    def forward(self, x):
        state = self.get_state(x)
        products = state.conj().unsqueeze(-2) * state[..., self.flips]
        ex = products.real.sum(-1)
        ey = (products * (-1j * self.signs)).real.sum(-1)
        ez = (state.abs().square().unsqueeze(-2) * self.signs).sum(-1)
        return torch.stack((ex, ey, ez), -1).flatten(-2).to(x.dtype)

    def masked_parameters(self, task):
        return [(self.weights, self.masks[task])]


class MLPBottleneck(nn.Module):
    def __init__(self, n_qubits=8, n_tasks=2, hidden_dim=96, **_):
        super().__init__()
        self.feature_dim, self.readout_dim = 2**n_qubits, 3*n_qubits
        self.fc1 = nn.Linear(self.feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, self.readout_dim)
        self.register_buffer("masks", partition(self.fc1.weight.shape, 0, n_tasks))

    def get_state(self, x):
        return F.gelu(self.fc1(x))

    def forward(self, x):
        return self.fc2(self.get_state(x))

    def masked_parameters(self, task):
        return [(self.fc1.weight, self.masks[task]), (self.fc1.bias, self.masks[task][:, 0])]


class OrthogonalBottleneck(nn.Module):
    def __init__(self, n_qubits=8, n_tasks=2, **_):
        super().__init__()
        self.feature_dim, self.readout_dim = 2**n_qubits, 3*n_qubits
        self.A = nn.Parameter(.01 * torch.randn(self.feature_dim, self.feature_dim))
        self.head = nn.Linear(self.feature_dim, self.readout_dim)
        blocks = partition((self.feature_dim,), 0, n_tasks)
        self.register_buffer("masks", blocks[:, :, None] & blocks[:, None, :])

    def matrix(self):
        return torch.matrix_exp((self.A - self.A.T).double()).to(self.A.dtype)

    def get_state(self, x):
        return x @ self.matrix().T

    def forward(self, x):
        return self.head(self.get_state(x))

    def masked_parameters(self, task):
        return [(self.A, self.masks[task])]


def build_bottleneck(cfg, tasks):
    classes = {"quantum": QuantumBottleneck, "classical_unconstrained": MLPBottleneck, "classical_orthogonal": OrthogonalBottleneck}
    return classes[cfg["bottleneck_type"]](n_tasks=tasks, **cfg)
