# TQSI continual SAM segmentation

Configuration-driven segmentation for local smoke tests and DGX A100 training. The model accepts RGB tensors `[B,3,H,W]` in `[0,1]` and returns raw mask logits `[B,K,H,W]`. Dataset discovery, labeling, splitting and augmentation stay outside the model.

Start with **[notebooks/TQSI_Training.ipynb](notebooks/TQSI_Training.ipynb)**. It trains task sequences and displays accuracy/loss/IoU/Dice, mIoU, boundary IoU, forgetting, t-SNE and image/ground-truth/prediction comparisons.

On this machine, the verified CUDA environment is `.venv-cuda/Scripts/python.exe` (Python 3.10, Torch 2.5.1+cu121). Select that interpreter as the notebook kernel. It reuses the existing local dependencies; it is not portable to DGX. The isolated `.venv/Scripts/python.exe` environment also passes the tests and uses CPU Torch. An executed notebook is saved locally at `outputs/notebook_executed.ipynb`.

**Status:** implementation and validation details are in [REPORT.md](REPORT.md). High accuracy and zero forgetting are experimental objectives, not guarantees. The supplied separation-loss premise is mathematically impossible for a shared unitary; see [specification review](docs/SPEC_REVIEW.md) before interpreting results.

## Install

Use Python 3.10–3.13 in an isolated environment. Install a matching Torch/torchvision CUDA build for the host using the [official PyTorch instructions](https://pytorch.org/get-started/locally/), then install this project:

```powershell
py -3.13 -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]"
.venv/Scripts/python.exe -m jupyterlab notebooks/TQSI_Training.ipynb
```

CPU testing can explicitly install `torch torchvision --index-url https://download.pytorch.org/whl/cpu` first. Choose the venv's Python kernel in VS Code/Jupyter. On DGX, create a Linux venv and install a driver-compatible CUDA Torch build first. The tested local dependency versions are recorded in `validation/` when available. The old Torch/PennyLane pins from the document are not forced onto newer Python versions.

## Run locally

From the repository root:

```powershell
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m tqsi.experiments toy --output outputs/toy
.venv/Scripts/python.exe -m tqsi.train --config configs/local_smoke.yaml
.venv/Scripts/python.exe -m tqsi.train --config configs/local_sam_smoke.yaml
```

`local_smoke` uses a small random frozen surrogate backbone, real dataset tiles and the real quantum bottleneck to test plumbing. `local_sam_smoke` uses the existing pretrained SAM ViT-B checkpoint and the full 8-qubit, 6-layer architecture, on a bounded real-data subset. It defaults to CPU for 4GB compatibility; change `device: cuda` to test the local GPU. Small source tiles are resized before SAM's 1024-input preprocessing; `image_size` does not reduce SAM's encoder grid. Full SAM inference can exceed 4GB depending on batch/replay size. Start with batch 1, replay batch 0 if memory is tight.

The old one-epoch `local_smoke` output is intentionally not an accuracy result: it used randomly selected tiles and collapsed to background. Current smoke configs use foreground-balanced source tiles, foreground-centred crops, BCE/focal/Tversky weighting, and a trainable spatial decoder projection over frozen encoder features. Each epoch reports foreground precision, recall and predicted foreground percentage beside IoU/Dice. `local_learning_debug.yaml` is a data/metric check only because its random frozen tiny backbone has no semantic feature signal. Run `configs/local_sam_learning_debug.yaml` on CUDA before DGX; do not begin a DGX sequence if it reports empty or full foreground predictions.

For binary datasets, the trainer calibrates a per-task logit threshold on validation Dice each epoch and saves that threshold with the validation-selected checkpoint. Test metrics and prediction artifacts use the saved threshold; test masks are never used for calibration.

Run the capacity gate before DGX continual learning:

```powershell
.venv-cuda/Scripts/python.exe -m tqsi.train --config configs/local_sam_overfit.yaml
```

The accuracy-first capacity gate is `configs/local_sam_capacity.yaml`. It audits raw labels, source geometry, foreground tile coverage and RGB/mask overlays before fitting eight foreground-balanced tiles. It must reach train-diagnostic Dice `0.95` before a benchmark run. `configs/dgx_sota_single_task.yaml` selects checkpoints by foreground validation IoU and the full 512px validation gate is `0.6277`; do not start continual learning until it passes.

Its `val` rows are deliberately marked `train_diagnostic`: they use the same eight balanced training tiles to test fitting capacity. A low Dice here means the model/adapter needs redesign. A high Dice only shows the path can fit those examples; then run the separate held-out real-SAM diagnostic and single-task DGX baseline.

Default dataset root: `D:/AI4CV_CL_DGX_A100/A100_datasets`. Default checkpoint: `D:/AI4CV_CL_DGX_A100/models/sam_vit_b_01ec64.pth`. No dataset or checkpoint is committed. The explicit `scripts/download_sam_checkpoint.py` helper downloads SAM if needed.

Each run requires a new `output_dir`, or an explicit task-boundary resume. If foreground Dice is undefined on an all-empty validation subset, selection falls back to validation loss and records that choice in the checkpoint:

```powershell
.venv/Scripts/python.exe -m tqsi.train --config configs/local_smoke.yaml --resume outputs/local_smoke/task_0_complete.pt
```

Only load trusted checkpoints. They contain Python/Torch state. Resume resets optimizer moments at the next task, matching uninterrupted training. Inside-task recovery is not implemented; resume the previous completed boundary to rerun the interrupted task.

## DGX A100

Edit `configs/dgx.yaml` for Linux dataset/checkpoint paths. It trains 24 epochs/task with replay enabled; `base.yaml` reproduces the specified no-replay variant. Batch size is **per GPU**, excluding extra replay examples. The implementation uses FP32 and Torch-differentiable complex state simulation. Start with one GPU to validate the environment before increasing to eight:

```bash
python -m tqsi.train --config configs/dgx.yaml
NPROC_PER_NODE=8 bash scripts/launch_dgx.sh configs/dgx.yaml
```

DDP shards training batches, synchronizes gradients, uses rank-zero validation/test evaluation and shared-directory checkpoints. Training samplers can pad a few examples when sample counts are not divisible by GPU count; validation/test never use padded distributed samplers. All ranks need the same dataset and output paths. Multi-GPU throughput and memory must be profiled on DGX; local tests cannot certify eight-GPU performance. `lightning.gpu` is not silently substituted for the differentiable state QNode.

Run `configs/dgx_sota_single_task.yaml` first. It uses the trainable FiLM spatial decoder on frozen SAM features; the quantum circuit supplies global conditioning. It is the accuracy configuration. `sam_prompt` remains the original frozen-SAM mask-decoder experiment and should be treated as an ablation. Only start `configs/dgx_sota_continual.yaml` after the single-task validation result is strong enough for the chosen target. The continual run rehearses compact labelled tiles and distils each stored tile's selected-model logits, explicitly protecting the shared decoder against functional forgetting.

## Add a dataset with one YAML

Copy `configs/datasets/example.yaml`, configure root/globs, optional stem suffixes, foreground labels or multiclass `label_map`, ignore labels and optional color mapping/group regex. Add its path to the run YAML's `tasks` list. Every task must share the same output ontology (`class_names`/`num_classes`). Non-raster annotations need an adapter before using this pipeline.

Sources are split **70% train / 20% validation / 10% test before tiling**. Largest-remainder rounding applies to group counts. At least 10 groups are required. LandCover.ai splits by source TIFF; OpenEarthMap splits by region. Grouped image/tile proportions may differ from group ratios. Existing upstream train/val/test files are intentionally not used because this is the requested custom split. Never compare these scores as if they used the upstream official benchmark split.

Both prepared dataset YAMLs extract buildings, preserving a consistent binary task across domains. OpenEarthMap label 0 is ignored; LandCover background 0 is valid. New multiclass sequences set the common model class count and map each dataset into a common ontology. Unmapped or invalid mask IDs fail explicitly.

## Outputs

```text
=== Stage 1/2: task-name ===
Training started at YYYY-MM-DD HH:MM:SS
Learning rate: ...
Trainable parameters: ...
Train Epoch 1: 100%|...| N/N [...]
Calibrate Epoch 1: 100%|...| N/N [...]
Eval Epoch 1: 100%|...| N/N [...]
Epoch [1/24] | train acc=... loss=... | val acc=... loss=... | Dice=... IoU=... mIoU=... BIoU=...
[Epoch 1] lr=... train_loss=... train_accuracy=... train_iou=... train_dice=... train_biou=... | val_loss=... val_accuracy=... val_iou=... val_dice=... val_biou=...
Validation source=validation | foreground P/R/pred=.../.../...%
Run epoch time: ...s
```

- `splits/*.json`: persisted source splits, fingerprints, unmatched-file audit.
- `resolved_config.json`, `datasets.json`, `environment.json`: provenance and parameter counts.
- `history.csv/json`, `learning_curves.png`: each task/epoch's metrics, losses and timing.
- `task_N_best.pt`: best validation-Dice model; `task_N_complete.pt`: continual recovery state.
- `after_task_N_test.json`, `test_iou_matrix.json`, `summary.json`: held-out metrics and forgetting.
- `diagnostics.json`, `overlap_stability.png`: current reference overlaps and old-task stability.
- `after_N_task_M_predictions.png/.npz`: visual comparisons and raw logits/masks.
- `tsne.png`, `tsne_features.npz`, `tsne_coordinates.npz`: held-out representation visualization.

Binary IoU/Dice/BIoU are foreground metrics; mIoU includes background. Absent-class metrics are undefined/null, not silently perfect. Loss is BCE+Dice for binary, CE+Dice for multiclass. Logged train/val loss is segmentation loss; separate columns record separation, stability, replay and replay-distillation losses. Forgetting follows the provided diagonal-minus-final formula and retains negative backward transfer.

## Research controls

```bash
python -m tqsi.experiments toy --output outputs/toy
python -m tqsi.experiments ladder --config configs/dgx.yaml --output outputs/ladder
python -m tqsi.experiments ablations --config configs/dgx.yaml --output configs/generated
python -m tqsi.experiments summarize --runs outputs/ablations --output outputs/ablation_report
python -m tqsi.evaluate --checkpoint outputs/dgx_quantum/task_1_complete.pt --device cuda
```

The ladder runs B1 MLP, B2 orthogonal and B3 quantum with three seeds and reports mean/std. Parameter counts are explicit and **not claimed matched**. Ablation generation writes executable configurations for depth, qubits, loss weights and task order with quantum/orthogonal controls; it does not launch expensive jobs automatically. A CA-SAM B0 reproduction and the reported SOTA reference are not fabricated or treated as measured baselines. The reference project's [dataset separation design](https://github.com/Arun2005-srm/skin-lesion-segmentation-refactored) informed the tensor contract; this repository implements the supplied TQSI architecture.
