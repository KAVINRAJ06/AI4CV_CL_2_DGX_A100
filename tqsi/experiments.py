"""Runnable mechanism controls, real-data baseline ladder and ablation manifests."""
import argparse
import copy
import csv
import itertools
from pathlib import Path
import time
import numpy as np
import torch
import yaml
from .artifacts import write_json, plotting
from .bottlenecks import build_bottleneck, normalize
from .config import load_config
from .continual import TaskController, masked_step, fidelity
from .train import run, seed_all


def toy(output="outputs/toy", steps=50, seeds=(0, 1, 2)):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows, trajectories = [], []
    for seed, kind, strategy, stability_weight in itertools.product(seeds,
            ("quantum", "classical_unconstrained", "classical_orthogonal"), ("qubit", "layer"), (0., .1)):
        seed_all(seed)
        cfg = dict(bottleneck_type=kind, n_qubits=4, n_layers=4, mask_strategy=strategy, hidden_dim=16)
        model = build_bottleneck(cfg, 2)
        controller = TaskController()
        inputs = normalize(torch.randn(2, 4, 16))
        targets = torch.randn(2, 4, 12)*.2
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.01)
        begin = time.perf_counter()
        for task in range(2):
            controller.register(task, inputs[task])
            for step in range(steps):
                fit = (model(inputs[task])-targets[task]).square().mean()
                sep, stab = controller.losses(model)
                masked_step(fit+.1*sep+stability_weight*stab, optimizer, model.masked_parameters(task), list(model.parameters()))
                if step % 10 == 0 or step == steps-1:
                    trajectories.append(dict(seed=seed, kind=kind, strategy=strategy, lambda_stab=stability_weight,
                        task=task, step=step, fit=float(fit.detach()), separation=float(sep.detach()),
                        stability_loss=float(stab.detach()), **controller.diagnostics(model)))
            controller.freeze(task, model)
        diag = controller.diagnostics(model)
        # Test the scientific invariant directly, rather than asserting an impossible decrease.
        states = model.get_state(inputs[:, 0])
        overlap = fidelity(states[0], states[1])
        before = fidelity(inputs[0, 0], inputs[1, 0])
        params = model.masked_parameters(0)[0][0]
        grad, = torch.autograd.grad(overlap, params)
        rows.append(dict(seed=seed, kind=kind, strategy=strategy, lambda_stab=stability_weight,
            parameter_count=sum(p.numel() for p in model.parameters()), seconds=time.perf_counter()-begin,
            overlap=diag["overlap"][0][1], old_task_stability=diag["stability"]["0"],
            invariant_error=float((overlap-before).abs().detach()), separation_gradient_max=float(grad.abs().max())))
    with (output / "baseline_table.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "trajectories.json", trajectories)
    write_json(output / "acceptance.json", dict(shared_unitary_separation_claim="mathematically impossible; invariant tested",
        parameter_matching="Not matched: exact counts in baseline_table.csv; dense B1/B2 exceed circuit count",
        quantum_advantage="not established", rows=rows))
    plt = plotting()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for kind in ("quantum", "classical_unconstrained", "classical_orthogonal"):
        for weight in (0., .1):
            selected = [r for r in trajectories if r["kind"] == kind and r["seed"] == seeds[0] and r["strategy"] == "qubit" and r["lambda_stab"] == weight and r["task"] == 1]
            axes[0].plot([r["separation"] for r in selected], label=f"{kind}, stab={weight}")
            axes[1].plot([r["stability"].get("0", np.nan) for r in selected])
    axes[0].set_title("Separation loss (common unitaries cannot change it)")
    axes[0].legend(fontsize=6)
    axes[1].set_title("Old-task state stability; controls included")
    fig.tight_layout()
    fig.savefig(output / "mechanism.png", dpi=140)
    plt.close(fig)
    return rows


def ladder(config, output, seeds=(0, 1, 2)):
    base = load_config(config)
    rows = []
    for kind, seed in itertools.product(("classical_unconstrained", "classical_orthogonal", "quantum"), seeds):
        cfg = copy.deepcopy(base)
        cfg["model"]["bottleneck_type"] = kind
        cfg["seed"] = seed
        cfg["output_dir"] = str(Path(output) / f"{kind}_seed{seed}")
        result = run(cfg)
        for routing in ("oracle", "router"):
            rows.append(dict(kind=kind, seed=seed, routing=routing, **{k: result[k] for k in ("last_iou", "avg_iou", "ff_iou")}))
        write_json(Path(output) / "ladder_raw.json", rows)
    summary = []
    for kind in sorted({r["kind"] for r in rows}):
        selected = [r for r in rows if r["kind"] == kind and r["routing"] == "router"]
        result = dict(kind=kind, seeds=len(selected))
        for metric in ("last_iou", "avg_iou", "ff_iou"):
            values = [r[metric] for r in selected]
            result[metric+"_mean"], result[metric+"_std"] = float(np.mean(values)), float(np.std(values))
        summary.append(result)
    write_json(Path(output) / "ladder_summary.json", summary)
    return rows


def generate_ablations(config, output):
    base = load_config(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    variants = []
    for depth in (1, 2, 4, 6, 8, 12):
        variants.append((f"depth_{depth}", {"n_layers": depth}, {}))
    for qubits in (6, 8, 10, 12):
        variants.append((f"qubits_{qubits}", {"n_qubits": qubits}, {}))
    for sep, stab in itertools.product((.01, .1, 1., 5.), repeat=2):
        variants.append((f"lambda_{sep}_{stab}", {}, dict(lambda_sep=sep, lambda_stab=stab)))
    for order in range(3):
        tasks = base["tasks"].copy()
        np.random.default_rng(order).shuffle(tasks)
        variants.append((f"order_{order}", {}, dict(tasks=tasks)))
    paths = []
    for name, model_update, update in variants:
        for kind in ("quantum", "classical_orthogonal"):
            cfg = copy.deepcopy(base)
            cfg.update(update)
            cfg["model"].update(model_update, bottleneck_type=kind)
            cfg["output_dir"] = f"outputs/ablations/{name}_{kind}"
            path = output / f"{name}_{kind}.yaml"
            path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            paths.append(str(path))
    write_json(output / "manifest.json", dict(configs=paths,
        note="Two tasks have only two unique orderings; three order seeds cannot create three distinct orders. Add a third task for that requirement."))
    return paths


def summarize_ablations(directory, output):
    """Collect measured run artifacts only; absent/failed experiments stay absent."""
    import json
    directory, output = Path(directory), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for summary_path in directory.glob("*/summary.json"):
        run_dir = summary_path.parent
        result = json.loads(summary_path.read_text())
        cfg = json.loads((run_dir / "resolved_config.json").read_text())
        history = json.loads((run_dir / "history.json").read_text())
        rows.append(dict(run=run_dir.name, kind=cfg["model"]["bottleneck_type"],
            n_layers=cfg["model"]["n_layers"], n_qubits=cfg["model"]["n_qubits"],
            lambda_sep=cfg["lambda_sep"], lambda_stab=cfg["lambda_stab"], seed=cfg["seed"],
            task_order=" -> ".join(cfg["tasks"]), epoch_seconds=float(np.mean([h["seconds"] for h in history])),
            **{k: result[k] for k in ("last_iou", "avg_iou", "ff_iou")}))
    if not rows:
        raise ValueError(f"No completed summary.json runs under {directory}")
    with (output / "measured_runs.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plt = plotting()
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for kind in sorted({r["kind"] for r in rows}):
        selected = [r for r in rows if r["kind"] == kind]
        for ax, prefix, xkey, ykey in ((axes[0, 0], "depth_", "n_layers", "last_iou"),
                                      (axes[0, 1], "qubits_", "n_qubits", "epoch_seconds"),
                                      (axes[1, 0], "lambda_", "ff_iou", "last_iou")):
            values = [r for r in selected if r["run"].startswith(prefix) and r[ykey] is not None and r[xkey] is not None]
            if values:
                ax.scatter([r[xkey] for r in values], [r[ykey] for r in values], label=kind)
            ax.set(xlabel=xkey, ylabel=ykey)
        values = [r["ff_iou"] for r in selected if r["run"].startswith("order_") and r["ff_iou"] is not None]
        if values:
            axes[1, 1].bar(kind, np.mean(values), yerr=np.std(values))
    axes[1, 1].set_title("Forgetting across completed order runs (mean/std)")
    for ax in axes.flat:
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "ablation_results.png", dpi=140)
    plt.close(fig)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("toy", "ladder", "ablations", "summarize"))
    parser.add_argument("--config", default="configs/local_smoke.yaml")
    parser.add_argument("--output", default="outputs/experiments")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--runs", default="outputs/ablations")
    args = parser.parse_args()
    if args.mode == "toy":
        toy(args.output, args.steps)
    elif args.mode == "ladder":
        ladder(args.config, args.output)
    elif args.mode == "ablations":
        generate_ablations(args.config, args.output)
    else:
        summarize_ablations(args.runs, args.output)


if __name__ == "__main__":
    main()
