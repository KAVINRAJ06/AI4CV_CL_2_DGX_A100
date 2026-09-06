"""Regenerate the checked-in notebook using only the Python standard library."""
import json
from pathlib import Path

cells = []
def md(text):
    cells.append(dict(cell_type="markdown", metadata={}, source=text.splitlines(True)))
def code(text):
    cells.append(dict(cell_type="code", metadata={}, source=text.splitlines(True), execution_count=None, outputs=[]))

md("""# TQSI continual segmentation
Run from the repository's Python environment. Select **local_smoke** for a fast tensor/continual-pipeline check, **local_sam_smoke** for the real frozen SAM + quantum architecture, or **dgx** for full training. Smoke metrics are not accuracy claims. Read `docs/SPEC_REVIEW.md` for the invariant separation loss and forgetting limitations.
""")
code("""from pathlib import Path
import os, sys, json
ROOT = Path.cwd()
if not (ROOT / 'tqsi').is_dir() and (ROOT.parent / 'tqsi').is_dir():
    ROOT = ROOT.parent
assert (ROOT / 'tqsi').is_dir(), 'Open this notebook from the repository'
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
import torch
print('Python:', sys.executable)
print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())
""")
md("""## Select the run
The defaults point to the provided local datasets. Change `dataset_root` and SAM checkpoint for DGX. Use one YAML per new dataset, and list them in task order. A fresh output directory prevents accidental overwriting of splits and results.
""")
code("""from datetime import datetime
from tqsi.config import load_config
PROFILE = 'local_smoke'  # 'local_sam_smoke' or 'dgx'
cfg = load_config(ROOT / 'configs' / f'{PROFILE}.yaml')
cfg['output_dir'] = str(ROOT / 'outputs' / f'notebook_{PROFILE}_{datetime.now():%Y%m%d_%H%M%S}')
# For CUDA smoke on the RTX 3050: cfg['device'] = 'cuda'
# On DGX: cfg['dataset_root'] = '/data/A100_datasets'
# cfg['model']['sam_checkpoint'] = str(ROOT / 'checkpoints/sam_vit_b_01ec64.pth')
print(json.dumps(cfg, indent=2))
""")
md("""## Train tasks sequentially
Each epoch prints train/validation accuracy, segmentation loss, IoU, Dice, mIoU, BIoU and elapsed time. Test evaluation occurs after validation selection, for all seen tasks. This cell uses one device. For eight GPUs launch `scripts/launch_dgx.sh` from a terminal (or the optional cell below).
""")
code("""from tqsi.train import run
summary = run(cfg)
summary
""")
md("""## Metrics, forgetting, predictions and t-SNE
Binary IoU/Dice/BIoU are foreground metrics; mIoU includes background. Forgetting is recorded as post-task IoU minus final IoU, excluding the last task from the average. t-SNE is a qualitative visualization of held-out features.
""")
code("""from IPython.display import display, Image
OUT = Path(cfg['output_dir'])
display(json.loads((OUT / 'summary.json').read_text()))
display(json.loads((OUT / 'test_iou_matrix.json').read_text()))
for name in ['learning_curves.png', 'overlap_stability.png', 'tsne.png']:
    if (OUT / name).exists():
        display(Image(filename=str(OUT / name)))
for path in sorted(OUT.glob('after_*_predictions.png')):
    display(Image(filename=str(path)))
""")
md("""## Optional: resume a completed task boundary
Checkpoints include model, references, replay samples, prototypes, split fingerprints and history. Resume restarts at the next task; interruption inside a task reruns that task from the prior completed boundary. Use the original config. Run directories and source manifests must remain available.
""")
code("""# checkpoint = OUT / 'task_0_complete.pt'
# summary = run(cfg, resume=checkpoint)
""")
md("""## Optional: toy controls, baseline ladder and DGX
These are explicit experiments; leave commented until needed. Full baseline comparisons require the same data protocol and at least three seeds. Oracle/router segmentation outputs coincide in this task-independent forward architecture; router task classification is reported separately.
""")
code("""# from tqsi.experiments import toy, ladder, generate_ablations
# toy(output='outputs/toy', steps=50)
# ladder('configs/dgx.yaml', output='outputs/ladder')
# generate_ablations('configs/dgx.yaml', 'configs/generated')
# import subprocess
# subprocess.run(['bash', 'scripts/launch_dgx.sh', 'configs/dgx.yaml'], check=True)
""")
notebook = dict(cells=cells, metadata=dict(kernelspec=dict(display_name="Python 3", language="python", name="python3"), language_info=dict(name="python", version="3.13")), nbformat=4, nbformat_minor=5)
for i, cell in enumerate(cells):
    cell["id"] = f"tqsi-{i:03d}"
path = Path("notebooks/TQSI_Training.ipynb")
path.parent.mkdir(exist_ok=True)
path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
