"""Isolate the actual broadcast quantum circuit, without datasets or SAM.

Run from the repository root: python -m scripts.profile_quantum --device cuda
"""
import argparse
import time

import torch
from tqsi.bottlenecks import QuantumBottleneck


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    if args.batch_size < 1 or args.steps < 1:
        parser.error("batch-size and steps must be positive")
    device = torch.device(args.device)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    model = QuantumBottleneck(n_qubits=8, n_layers=6, n_tasks=2).to(device)
    print(f"Torch {torch.__version__}; device={device}; batch={args.batch_size}; 8 qubits, 6 layers", flush=True)
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    for step in range(args.steps):
        model.zero_grad(set_to_none=True)
        x = torch.randn(args.batch_size, 256, device=device, requires_grad=True)
        sync()
        started = time.perf_counter()
        print(f"Step {step+1}: forward starting", flush=True)
        result = model(x)
        loss = result.square().mean()
        sync()
        forward = time.perf_counter()-started
        print(f"Forward {forward:.3f}s; backward starting", flush=True)
        started = time.perf_counter()
        loss.backward()
        sync()
        elapsed = time.perf_counter()-started
        assert torch.isfinite(x.grad).all() and torch.isfinite(model.weights.grad).all()
        print(f"Backward {elapsed:.3f}s; input grad norm={x.grad.norm().item():.6g}; weight grad norm={model.weights.grad.norm().item():.6g}", flush=True)


if __name__ == "__main__":
    main()
