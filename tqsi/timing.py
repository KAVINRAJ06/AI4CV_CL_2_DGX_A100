"""Flushed block timings, synchronized on the selected CUDA device."""
from contextlib import contextmanager
import time

import torch


class BlockTimer:
    def __init__(self, device="cpu", enabled=False):
        self.device = torch.device(device)
        self.enabled = enabled

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @contextmanager
    def block(self, label):
        if not self.enabled:
            yield
            return
        self.synchronize()
        print(f"[Timing] START {label}", flush=True)
        started = time.perf_counter()
        try:
            yield
            self.synchronize()
        except BaseException:
            print(f"[Timing] FAILED {label} | {time.perf_counter()-started:.3f}s", flush=True)
            raise
        else:
            print(f"[Timing] END {label} | {time.perf_counter()-started:.3f}s", flush=True)

    def batches(self, batches, label):
        with self.block(f"{label}: create data iterator"):
            iterator = iter(batches)
        for index in range(len(batches)):
            with self.block(f"{label}: load batch {index+1}/{len(batches)}"):
                batch = next(iterator)
            yield batch
