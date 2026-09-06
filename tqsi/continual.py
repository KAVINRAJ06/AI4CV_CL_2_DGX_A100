import torch
from torch.nn import functional as F
from .bottlenecks import normalize


def fidelity(a, b):
    a, b = normalize(a), normalize(b)
    return (a.conj() * b).sum(-1).abs().square().real


def segmentation_loss(logits, target):
    valid = target != -100
    if not valid.any():
        return logits.sum() * 0
    if logits.shape[1] == 1:
        truth = target.clamp_min(0).float().unsqueeze(1)
        prob = logits.sigmoid()
        raw = F.binary_cross_entropy_with_logits(logits, truth, reduction="none")
        data_loss = raw[:, 0][valid].mean()
    else:
        truth = F.one_hot(target.clamp_min(0), logits.shape[1]).permute(0, 3, 1, 2).float()
        prob = logits.softmax(1)
        data_loss = F.cross_entropy(logits, target, ignore_index=-100)
    prob, truth = prob * valid[:, None], truth * valid[:, None]
    dims = (0, 2, 3)
    dice = (2*(prob*truth).sum(dims)+1e-6)/(prob.sum(dims)+truth.sum(dims)+1e-6)
    return data_loss + 1-dice.mean()


class TaskController:
    def __init__(self):
        self.inputs, self.frozen = {}, {}

    def register(self, task, inputs):
        self.inputs[task] = inputs.detach().cpu().clone()

    def live(self, bottleneck):
        device = next(bottleneck.parameters()).device
        return {t: bottleneck.get_state(x.to(device)) for t, x in self.inputs.items()}

    def losses(self, bottleneck):
        states = self.live(bottleneck)
        zero = next(bottleneck.parameters()).sum()*0
        ids = list(states)
        terms = [fidelity(states[a].mean(0), states[b].mean(0)) for i, a in enumerate(ids) for b in ids[i+1:]]
        sep = torch.stack(terms).mean() if terms else zero
        stab = [1-fidelity(ref.to(states[t].device), states[t]).mean() for t, ref in self.frozen.items()]
        return sep, torch.stack(stab).mean() if stab else zero

    @torch.no_grad()
    def freeze(self, task, bottleneck):
        self.frozen[task] = self.live(bottleneck)[task].detach().cpu()

    @torch.no_grad()
    def diagnostics(self, bottleneck):
        states = self.live(bottleneck)
        ids = list(states)
        overlap = [[float(fidelity(states[a].mean(0), states[b].mean(0))) for b in ids] for a in ids]
        stability = {str(t): float(fidelity(ref.to(states[t].device), states[t]).mean()) for t, ref in self.frozen.items()}
        return dict(tasks=ids, overlap=overlap, stability=stability)

    def state_dict(self):
        return dict(inputs=self.inputs, frozen=self.frozen)

    def load_state_dict(self, state):
        self.inputs, self.frozen = state["inputs"], state["frozen"]


def masked_step(loss, optimizer, masked, parameters, clip=1.):
    """Protect against Adam momentum AND AdamW decay, not just gradient leakage."""
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    backups = []
    for parameter, mask in masked:
        if parameter.grad is None:
            raise RuntimeError("Masked parameter has no gradient")
        backups.append(parameter.detach().clone())
        parameter.grad.mul_(mask)
        for value in optimizer.state.get(parameter, {}).values():
            if isinstance(value, torch.Tensor) and value.shape == parameter.shape:
                value.mul_(mask)
    torch.nn.utils.clip_grad_norm_(parameters, clip, error_if_nonfinite=True)
    optimizer.step()
    with torch.no_grad():
        for (parameter, mask), before in zip(masked, backups):
            parameter[~mask] = before[~mask]
            for value in optimizer.state.get(parameter, {}).values():
                if isinstance(value, torch.Tensor) and value.shape == parameter.shape:
                    value.mul_(mask)
            assert torch.equal(parameter[~mask], before[~mask])
