from pathlib import Path
import csv
import json
import math
import numpy as np
import torch


def write_json(path, data):
    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (tuple, list)):
            return [clean(v) for v in x]
        if isinstance(x, float) and not math.isfinite(x):
            return None
        return x
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(data), indent=2), encoding="utf-8")
    temporary.replace(path)


def plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def history_artifacts(directory, history, diagnostics):
    directory = Path(directory)
    write_json(directory / "history.json", history)
    write_json(directory / "diagnostics.json", diagnostics)
    if not history:
        return
    with (directory / "history.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    plt = plotting()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, metric in zip(axes.flat, ("accuracy", "loss", "iou", "dice", "miou", "biou")):
        for split in ("train", "val"):
            ax.plot([h[f"{split}_{metric}"] for h in history], label=split)
        ax.set(title=metric, xlabel="Epoch across task sequence")
        ax.legend()
    fig.tight_layout()
    fig.savefig(directory / "learning_curves.png", dpi=140)
    plt.close(fig)
    if diagnostics:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].imshow(diagnostics[-1]["overlap"], vmin=0, vmax=1)
        axes[0].set(title="Current reference state fidelity", xlabel="Task", ylabel="Task")
        for task in diagnostics[-1]["stability"]:
            axes[1].plot([d["stability"].get(task, np.nan) for d in diagnostics], label=f"task {task}")
        axes[1].set(title="Frozen reference stability", xlabel="Epoch across tasks", ylim=(0, 1.01))
        if diagnostics[-1]["stability"]:
            axes[1].legend()
        fig.tight_layout()
        fig.savefig(directory / "overlap_stability.png", dpi=140)
        plt.close(fig)


@torch.no_grad()
def predictions(model, loader, path, device, count=4):
    plt = plotting()
    images, truth = next(iter(loader))
    images, truth = images[:count], truth[:count]
    logits = model(images.to(device)).cpu()
    pred = (logits[:, 0] > 0).long() if logits.shape[1] == 1 else logits.argmax(1)
    mask_cmap = plt.get_cmap("gray" if model.classes == 1 else "tab20").copy()
    mask_cmap.set_bad("#888888")
    fig, axes = plt.subplots(len(images), 3, figsize=(10, 3*len(images)), squeeze=False)
    for i in range(len(images)):
        for ax, value, title in zip(axes[i], (images[i].permute(1, 2, 0), truth[i], pred[i]), ("Image", "Ground truth", "Prediction")):
            if title == "Image":
                ax.imshow(value)
            else:
                value = np.ma.masked_where(np.asarray(value) == -100, np.asarray(value))
                ax.imshow(value, cmap=mask_cmap, vmin=0, vmax=max(1, model.classes-1), interpolation="nearest")
            ax.set_title(title)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    np.savez_compressed(Path(path).with_suffix(".npz"), images=images.numpy(), targets=truth.numpy(), predictions=pred.numpy(), logits=logits.numpy())


@torch.no_grad()
def tsne(model, loaders, directory, device, seed=42, limit=128):
    features, labels = [], []
    for task, loader in enumerate(loaders):
        count = 0
        for x, _ in loader:
            x = x[:limit-count].to(device)
            if not len(x):
                break
            features.append(model.representation(x).cpu().numpy())
            labels.extend([task]*len(x))
            count += len(x)
            if count >= limit:
                break
    x = np.concatenate(features)
    np.savez_compressed(Path(directory) / "tsne_features.npz", features=x, tasks=labels)
    if len(x) < 4:
        write_json(Path(directory) / "tsne_status.json", {"status": "skipped", "reason": "At least 4 held-out samples needed"})
        return
    from sklearn.manifold import TSNE
    xy = TSNE(n_components=2, perplexity=min(30, len(x)-1), init="random", learning_rate="auto", random_state=seed).fit_transform(x)
    np.savez_compressed(Path(directory) / "tsne_coordinates.npz", coordinates=xy, tasks=labels)
    plt = plotting()
    fig, ax = plt.subplots(figsize=(7, 6))
    for task in sorted(set(labels)):
        selected = np.array(labels) == task
        ax.scatter(xy[selected, 0], xy[selected, 1], label=f"Task {task}", s=14)
    ax.legend()
    ax.set_title("Held-out projected features (t-SNE; qualitative only)")
    fig.tight_layout()
    fig.savefig(Path(directory) / "tsne.png", dpi=140)
    plt.close(fig)
