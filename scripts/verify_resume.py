"""Check task-boundary recovery against an already completed deterministic smoke run."""
import argparse
import json
from pathlib import Path
import torch
import numpy as np
from tqsi.train import run, load_checkpoint
from tqsi.artifacts import write_json

parser = argparse.ArgumentParser()
parser.add_argument("--source", default="outputs/local_smoke_verified")
parser.add_argument("--output", default="outputs/resume_verified")
args = parser.parse_args()
source, output = Path(args.source), Path(args.output)
config = json.loads((source / "resolved_config.json").read_text())
config["output_dir"] = str(output)
result = run(config, resume=source / "task_0_complete.pt")
original = load_checkpoint(source / "task_1_complete.pt")
resumed = load_checkpoint(output / "task_1_complete.pt")
equal = all(torch.equal(original["model"][key], resumed["model"][key]) for key in original["model"])
write_json(output / "resume_verification.json", dict(model_bit_identical=equal,
    same_test_iou_matrix=bool(np.array_equal(original["matrix"], resumed["matrix"], equal_nan=True)), summary=result))
assert equal, "Resumed model differs from uninterrupted run"
print("Resume produced bit-identical final model")
