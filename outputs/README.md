# Recorded local outputs

These files are the actual local runs, including metrics, plots, prediction tensors, split manifests, checkpoints and the executed notebook. Model checkpoints use Git LFS; run `git lfs pull` after cloning to retrieve their contents.

- `local_sam_smoke`: completed real pretrained SAM + quantum smoke on the RTX 3050, two training samples per task. Foreground Dice/IoU were zero; this does not meet the accuracy target.
- `local_smoke_verified`: completed surrogate-backbone pipeline smoke with replay, 16 samples per split/task.
- `resume_verified`: recovery check against the completed surrogate run. `validation/resume.json` contains the corrected NaN-aware matrix comparison; the earlier local resume report used ordinary list equality and reports a false mismatch for NaN placeholders.
- `notebook_local_smoke_*`: completed runs from notebook verification.
- `notebook_executed.ipynb`: executed notebook with displayed results.
- `toy`: measured quantum/classical mechanism controls, one seed and 20 steps/task.
- `local_smoke`: incomplete initial run that exposed undefined foreground Dice checkpoint selection. Kept as historical output; the bug was fixed subsequently.
- `sam_acceptance.json`: real SAM gradient, frozen-parameter and tensor-shape acceptance results.

These are bounded smoke tests, not DGX training or evidence of high accuracy or zero forgetting. Recorded run configurations and local source paths describe the machine on which each run was performed. See the root `REPORT.md` for interpretation.
