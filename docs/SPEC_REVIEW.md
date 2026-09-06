# Engineering decisions and scientific limits

The two supplied specifications and Python scaffold describe the desired architecture. They are design inputs, not permission to claim experiment results or to replace the user's 70/20/10 split with another benchmark split.

## Implemented architecture

Frozen pretrained SAM image encoder -> global pooling + Linear + LayerNorm + unit normalization -> amplitude encoding -> trainable, masked Rot/CNOT circuit -> Pauli X/Y/Z readout -> learned dense prompt projection -> frozen SAM decoder -> tensor logits. All circuit weights participate in every forward pass; task masks restrict updates only. Encoder and prompt encoder are frozen; decoder weights are frozen but decoder operations remain differentiable with respect to the learned prompt. The implementation uses the documented dense-prompt slot of original SAM, not a verified reproduction of CA-SAM's alignment layer.

Multiple semantic classes use one learned dense prompt per class and reuse the same frozen mask decoder, with CE + multiclass Dice. This is a stated extension; the original specification is binary (BCE + Dice). Default experiments map both local datasets to building/background.

The initial global-only decoder projection could only broadcast a 3n-dimensional readout over every spatial location. That is insufficient to learn small aerial objects with a frozen decoder. The configured decoder projection head therefore has two trainable components: readout-to-global prompt and a 1×1 spatial projection of the frozen encoder feature map. Their sum is the dense SAM prompt. Disable `model.decoder_spatial_adapter` only for the global-only ablation; it is expected to collapse on the local tiny surrogate.

## Impossible separation acceptance criterion

For fixed references a,b and the same unitary U, `<Ua, Ub> = <a, U*Ub> = <a,b>`. Therefore squared state fidelity is invariant and its gradient with respect to circuit weights is zero. Qubit/layer gradient masks do not change that identity. A common orthogonal matrix has the same property. The specified loss is preserved and logged, and an executable regression test checks invariance. We do not claim that changing entanglement, depth or lambda_sep can reduce this overlap. Multiple reference means remain invariant under the same linear unitary too.

Making separation learnable would require changing the model, e.g. task-dependent forward operators or trainable reference inputs, and would no longer be this exact shared-current-weight experiment. No such change is hidden in this implementation.

## Forgetting

Disjoint trainable parameter regions do not guarantee unchanged predictions: every forward uses current full weights, CNOTs couple wires, and projection/decoder heads are shared. Stability on a finite reference set does not guarantee stability on a full task. The optional, explicitly configured rehearsal buffer replays labeled TRAIN tiles to constrain the actual segmentation function, including shared heads. `base.yaml` disables replay to preserve the original protocol; `dgx.yaml` enables it as an extension. No zero-forgetting guarantee is made.

The router classifies domains using nearest frozen-encoder prototypes built from training samples. Oracle/router segmentation outputs are exactly identical under the specified task-independent forward. Domain classification accuracy is reported separately. Task-conditional adapters could make routing affect segmentation, but would change the architecture.

## Corrections to example code

- Adam/AdamW momentum and weight decay can move masked-off entries despite zero gradients. Updates restore protected values exactly and clear their optimizer state.
- Parameter partitions reject tasks exceeding available qubits/layers/hidden units.
- Circuit broadcasting processes batches in one QNode. Default entangling range is explicitly 1 in every layer.
- `default.qubit` with Torch backprop supports amplitude-input gradients and differentiable state outputs on CPU/CUDA. The specified `lightning.gpu` + `diff_method='backprop'` is not a supported drop-in replacement. No unverified Lightning backend is offered. DGX DDP distributes samples/classical modules and their local small circuit computations.
- Reference tensors move to the current device; zero losses stay attached to the correct device/graph.
- SAM's public forward is decorated no_grad. Training calls frozen encoder and decoder modules directly, preserves decoder input gradients, and follows original per-image decoding semantics.
- Full FP32 classical / complex64 quantum execution is the conservative default. There is no untested AMP claim.
- Dense B1/B2 parameter counts do not match the tiny circuit: the supplied hidden_dim=96 does not yield 144 parameters. Actual counts are reported. At 256 input and 24 output dimensions, even width-one dense B1 needs 305 parameters. A +/-10% matched dense comparison to a 144-parameter circuit is impossible as written. No matched-capacity or quantum-advantage claim is made.

## Dataset protocol

70/20/10 uses largest-remainder rounding on source groups, then tiles each partition independently. LandCover groups are original TIFFs; OpenEarthMap groups are regions. Group counts follow 70/20/10 (up to integer rounding); image/tile counts differ when group sizes differ. This deliberately differs from upstream benchmark splits and must not be presented as a direct CA-SAM benchmark comparison. Manifest audit records unmatched OpenEarthMap wo_xBD labels. Raster image dimensions, pair uniqueness and label values are validated. Source files are never modified.

Tiles on the image edge are resized to the configured square; metrics describe those evaluation tiles, not stitched geospatial maps. Files must be paired RGB raster images and indexed or configured-color raster masks. Arbitrary detection/COCO polygons/volumes/multispectral inputs require a raster adapter first; “new dataset by YAML” applies to this documented tensor contract.

## Metrics

Dataset-level confusion matrices accumulate pixels. Binary IoU/Dice/BIoU refer to foreground; mIoU averages background and foreground. Multiclass uses macro averages over classes with nonzero denominators. Absent classes are excluded (undefined becomes JSON null); accuracy excludes ignore pixels. Boundary IoU uses a square erosion radius round(0.02 * image diagonal), min 1, including image edges, excluding neighborhoods of ignore pixels. Report this convention when comparing to another implementation.

After each task, the validation-Dice-selected model evaluates all seen test splits. Last-IoU = mean final row; Avg-IoU = mean diagonal; FF-IoU = mean(diagonal - final row) over all but the last task. Negative forgetting is retained as backward transfer; one task returns zero. Test metrics do not select weights or tune thresholds.

## Unverified research claims

The quoted accuracy .9040, Dice .7448, IoU .6277, BIoU .2740 is a target reference only. Its dataset/protocol and interpretation of “mIoU” are unspecified. Full 24-epoch DGX experiments, three-seed B1/B2/B3 comparisons, CA-SAM reproduction, and ablation conclusions require actual runs. Scripts generate these experiments; source code is not evidence of high accuracy or quantum advantage. With two domains only two unique task orders exist, even if three order seeds are used.

## Sources inspected

- Dataset-contract reference: https://github.com/Arun2005-srm/skin-lesion-segmentation-refactored
- SAM decoder API: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/mask_decoder.py
- SAM preprocessing/decoding: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/sam.py
- LandCover label IDs: https://landcover.ai.linuxpolska.com/
- OpenEarthMap ontology: https://open-earth-map.org/overview_oem.html
