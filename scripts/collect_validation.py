"""Save compact measured evidence without publishing datasets or checkpoints."""
from pathlib import Path
import importlib.metadata
import json
import shutil
import numpy as np
from tqsi.train import load_checkpoint
from tqsi.artifacts import write_json

out = Path('validation')
out.mkdir(exist_ok=True)
for source, target in [
    ('outputs/local_sam_smoke/history.csv', 'sam_smoke_history.csv'),
    ('outputs/local_sam_smoke/summary.json', 'sam_smoke_summary.json'),
    ('outputs/local_sam_smoke/environment.json', 'sam_smoke_environment.json'),
    ('outputs/toy/baseline_table.csv', 'toy_baseline_table.csv'),
    ('outputs/sam_acceptance.json', 'sam_acceptance.json'),
]:
    shutil.copyfile(source, out / target)
debug = Path('outputs/local_sam_learning_debug_v3')
if debug.exists():
    shutil.copyfile(debug / 'history.csv', out / 'sam_learning_debug_history.csv')
    shutil.copyfile(debug / 'summary.json', out / 'sam_learning_debug_summary.json')
    shutil.copyfile(debug / 'after_task_0_test.json', out / 'sam_learning_debug_test.json')
original = load_checkpoint('outputs/local_smoke_verified/task_1_complete.pt')
resumed = load_checkpoint('outputs/resume_verified/task_1_complete.pt')
import torch
write_json(out / 'resume.json', dict(model_bit_identical=all(torch.equal(original['model'][k], resumed['model'][k]) for k in original['model']),
    test_matrix_equal=bool(np.array_equal(original['matrix'], resumed['matrix'], equal_nan=True))))
versions = {name: importlib.metadata.version(name) for name in ('torch', 'torchvision', 'pennylane', 'numpy', 'Pillow', 'PyYAML', 'matplotlib', 'scikit-learn', 'segment-anything', 'tifffile', 'pytest', 'nbclient', 'nbformat')}
write_json(out / 'tested_versions.json', versions)
notebook = json.loads(Path('outputs/notebook_executed.ipynb').read_text(encoding='utf-8'))
write_json(out / 'notebook.json', dict(executed=True, code_cells=sum(c['cell_type']=='code' for c in notebook['cells']),
    error_outputs=sum(o.get('output_type')=='error' for c in notebook['cells'] for o in c.get('outputs', []))))
print('Saved measured evidence to validation/')
