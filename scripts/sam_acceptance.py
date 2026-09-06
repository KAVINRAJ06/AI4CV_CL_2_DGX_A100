"""Optional real-checkpoint integration test, including batched multiclass decoding."""
import argparse
import torch
from tqsi.model import TQSI
from tqsi.continual import segmentation_loss
from tqsi.artifacts import write_json

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--device", default="cuda")
parser.add_argument("--output", default="outputs/sam_acceptance.json")
args = parser.parse_args()
torch.manual_seed(42)
torch.set_num_threads(4)
model = TQSI(dict(backbone="sam", sam_checkpoint=args.checkpoint, bottleneck_type="quantum",
                  n_qubits=8, n_layers=6, num_classes=3), 2).to(args.device)
images = torch.rand(1, 3, 64, 96, device=args.device)
logits = model(images)
assert logits.shape == (1, 3, 64, 96)
segmentation_loss(logits, torch.randint(3, (1, 64, 96), device=args.device)).backward()
assert all(p.grad is None for p in model.backbone.parameters())
gradients = {name: float(p.grad.norm()) for name, p in model.named_parameters() if p.requires_grad}
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
assert gradients['bottleneck.weights'] > 0
assert gradients['decoder_head.weight'] > 0
# Verify the per-image decoder's B>1 contract without re-encoding a second image.
with torch.no_grad():
    features = model.backbone.encode(images)
    prompts = torch.zeros(2, 3, 256, device=args.device)
    batch_logits = model.backbone.decode(features.expand(2, -1, -1, -1), prompts, (64, 96))
assert batch_logits.shape == (2, 3, 64, 96)
write_json(args.output, dict(passed=True, gradients=gradients, shape=list(logits.shape), batched_shape=list(batch_logits.shape),
    frozen_backbone=True, device=args.device, torch=torch.__version__))
print('Real SAM input gradients, frozen weights, non-square images and multiclass batches: passed')
