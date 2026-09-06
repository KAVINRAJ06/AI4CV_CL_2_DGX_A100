# Implementation and local validation — 2026-09-06

The requested tensor-only TQSI architecture, dataset pipeline, notebook, continual-training loop and evaluation artifacts are implemented. **The quoted SOTA accuracy/Dice/IoU target has not been achieved. Zero forgetting and quantum advantage have not been demonstrated.** Full DGX experiments remain to be run on DGX.

## Verified locally

- 15 regression tests passed in the existing Python 3.10 environment with Torch 2.5.1+cu121 and PennyLane 0.40.0.
- The same 15 tests passed in the new isolated Python 3.13 CPU environment with Torch 2.14.0+cpu and PennyLane 0.45.1. Exact versions for both tested environments are saved in `validation/tested_versions*.json`.
- Real SAM ViT-B + 8 qubits + 6 circuit layers completed one epoch on each of the two local domains on the RTX 3050 4GB. Each task used 2 train, 2 validation and 2 test tiles at 256 pixels, batch 1. Replay was disabled for this GPU smoke test. This is only four training updates across two tasks, not a fair 24-epoch accuracy experiment.
- Real SAM acceptance checks passed: non-square input `[1,3,64,96]`, 3-class logits, batch-of-two decoder output, finite nonzero trainable gradients, no gradients on frozen SAM weights.
- The full notebook executed, including training and result-display cells. The executed notebook remains local at `outputs/notebook_executed.ipynb`; the committed notebook has clean outputs.
- A separate 16/16/16-tile per-task run exercised replay, held-out evaluation and visual artifacts. Resuming after task 0 produced a bit-identical final model and equal test-IoU matrix.
- Toy quantum/MLP/orthogonal controls ran with and without stability regularization, for 20 steps/task, one seed. These are measured diagnostics, not completed three-seed research comparisons.

Compact numerical evidence is in `validation/`. Raw checkpoints, prediction images/logits, split manifests and plots stay in the ignored local `outputs/` directories.

## Real SAM smoke metrics

| Task | Val accuracy | Val loss | Foreground IoU | Dice | mIoU | BIoU | Epoch seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| LandCover buildings | 0.9638 | 1.3955 | 0.0000 | 0.0000 | 0.4819 | 0.0000 | 9.26 |
| OpenEarthMap buildings | 0.6000 | 5.0187 | 0.0000 | 0.0000 | 0.3000 | 0.0000 | 5.41 |

Last-IoU, Avg-IoU and FF-IoU were all 0. The zero forgetting value is uninformative because foreground segmentation was unsuccessful. High background pixel accuracy must not be mistaken for good segmentation. These bounded timings exclude initialization, post-task test evaluation and plotting, and cannot be compared to the supplied 196.75-second full epoch.

## Scientific result from the toy check

The supplied shared-unitary separation loss cannot train separation: `<Ua,Ub> = <a,b>`. The measured maximum separation-gradient magnitudes were about 1e-8 for the quantum/orthogonal controls. No monotonic overlap decrease can be honestly asserted under this architecture.

For the one-seed qubit-mask toy, old-task fidelity was 0.7207 without stability and 0.8744 with weight 0.1 for the quantum circuit. The orthogonal control achieved 0.7352 and 0.9752 respectively. These finite-reference toy values do not establish segmentation forgetting performance or a quantum advantage. Actual trainable bottleneck counts were 48 quantum, 476 MLP and 460 orthogonal; the example is not capacity-matched. See [SPEC_REVIEW.md](docs/SPEC_REVIEW.md) for why the document's dense matched-capacity configuration is inconsistent.

## Dataset audit

LandCover.ai: 41 paired original TIFFs, divided into 29 train / 8 validation / 4 test sources before tiling.

OpenEarthMap: 2,687 valid image/mask pairs over 75 regions. Regions split into 53 / 15 / 7 (70/20/10 with largest-remainder rounding), yielding 1,837 / 554 / 296 source images. The local tree additionally contains 1,151 images without masks and 813 masks without images. These are excluded and individually recorded in the manifest audit. No unavailable labels are inferred.

The split is independent of training seed (`split_seed: 42`) to keep baseline initialization comparisons on the same partitions. The custom source/region split intentionally differs from published benchmark splits.

## Remaining experiments and limits

- Run the full 24-epoch single-task SAM experiment and assess validation Dice/IoU before using continual-learning results to support accuracy claims.
- Run B1/B2/B3 on the same splits with three seeds, both without replay and with the explicitly labeled replay extension. B0 CA-SAM reproduction and its exact alignment-layer integration are not implemented as a verified baseline.
- Execute generated depth/qubit/loss-weight/order ablations and collect measured outputs with `tqsi.experiments summarize`. Three distinct task orders require at least three domains; the two available domains have only two orders.
- Profile and validate eight-GPU DDP on DGX. Its launcher/training code is implemented; this machine cannot verify the multi-GPU execution path. FP32 and `default.qubit` Torch backprop are supported paths; Lightning GPU/AMP are not claimed implemented.
- Task ID changes gradient allocation only, so oracle/router segmentation outputs are identical by construction. The domain router's classification accuracy is separately reported.
- Raster/label adapters support YAML-defined paired RGB segmentation datasets. Polygon annotations, volumes and multispectral imagery need preprocessing adapters.

The implementation does not fabricate results to satisfy impossible phase gates in the documents. It preserves those mechanisms, tests their actual properties, and records the deviations and extensions explicitly.
