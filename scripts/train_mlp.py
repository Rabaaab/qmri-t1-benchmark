"""Train the per-voxel MLP and score it against the classical baselines.

Run with:

    python scripts/build_dataset.py        # if you have not already
    python scripts/fit_baseline.py --split test
    python scripts/train_mlp.py

Writes ``outputs/mlp.pt``, ``outputs/mlp_training.png`` and
``outputs/mlp_vs_classical.png``, and prints the per-tissue comparison.

What this run is for: the MLP sees exactly what the classical fit sees --
one voxel's five measurements plus B1, and no neighbours. So the gap between
it and the corrected fit isolates the contribution of *learning*, with
spatial context held at zero. When the U-Net arrives, whatever it gains
beyond this line is the contribution of context, which is also the mechanism
that can erase a lesion.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from qmri.dataset import Dataset, scatter_to_maps, to_voxel_table
from qmri.models import VoxelMLP, to_ms, to_scaled
from qmri.tissues import LESION_3T, TISSUES_3T
from qmri.training import TrainConfig, train

TISSUE_NAMES = {t.label: t.name for t in TISSUES_3T if t.t1_ms > 0}
TISSUE_NAMES[LESION_3T.label] = LESION_3T.name


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data", type=Path, default=ROOT / "data")
    p.add_argument("--out", type=Path, default=ROOT / "outputs")
    return p.parse_args(argv)


def stats(estimate, truth, where):
    """Bias and scatter over a region, ignoring voxels a method gave up on."""
    error = estimate[where] - truth[where]
    error = error[np.isfinite(error)]
    if error.size == 0:
        return float("nan"), float("nan")
    return float(error.mean()), float(error.std())


def main(argv=None) -> int:
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    for name in ("train", "val", "test"):
        if not (args.data / f"{name}.npz").exists():
            print(f"missing {args.data / f'{name}.npz'}; run build_dataset.py")
            return 1

    train_set = Dataset.load(args.data / "train.npz")
    val_set = Dataset.load(args.data / "val.npz")
    test_set = Dataset.load(args.data / "test.npz")

    train_x, train_y_ms, _ = to_voxel_table(train_set)
    val_x, val_y_ms, _ = to_voxel_table(val_set)
    test_x, test_y_ms, test_mask = to_voxel_table(test_set)

    print(f"train {train_x.shape[0]:,} voxels, {train_x.shape[1]} features")
    print(f"val   {val_x.shape[0]:,} voxels")
    print(f"test  {test_x.shape[0]:,} voxels\n")

    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        seed=args.seed,
    )
    model, history = train(
        lambda: VoxelMLP(
            n_features=train_x.shape[1], hidden=args.hidden, depth=args.depth
        ),
        train_x,
        to_scaled(train_y_ms),
        val_x,
        to_scaled(val_y_ms),
        config,
    )
    print(
        f"\nbest epoch {history.best_epoch + 1} of {history.epochs_run}, "
        f"val loss {history.best_val_loss:.6f}"
    )

    torch.save(
        {"state_dict": model.state_dict(), "config": vars(args)},
        args.out / "mlp.pt",
    )

    # --- predict the test split ------------------------------------------
    with torch.no_grad():
        predicted = model.predict_ms(torch.as_tensor(test_x)).numpy()
    mlp_maps = scatter_to_maps(predicted, test_mask)

    # --- compare against the cached classical fits ------------------------
    baseline_path = args.data / "baseline_test.npz"
    if not baseline_path.exists():
        print(
            f"\nno {baseline_path.name}; run "
            "scripts/fit_baseline.py --split test for the comparison"
        )
        return 0

    with np.load(baseline_path) as f:
        naive, corrected = f["naive"], f["corrected"]

    truth = test_set.t1
    methods = [
        ("fit, no B1", naive),
        ("fit, with B1", corrected),
        ("MLP", mlp_maps),
    ]

    print()
    header = f"{'tissue':<14}{'true T1':>9}"
    for name, _ in methods:
        header += f"{name + ' bias':>18}{'SD':>8}"
    print(header)
    print("-" * len(header))

    for label, tissue_name in TISSUE_NAMES.items():
        where = test_set.labels == label
        if not where.any():
            continue
        row = f"{tissue_name:<14}{truth[where].mean():>9.0f}"
        for _, estimate in methods:
            bias, sd = stats(estimate, truth, where)
            row += f"{bias:>18.1f}{sd:>8.1f}"
        print(row)

    print()
    print("bias and SD in ms, over the whole test split.")
    print(
        "The 'lesion' row is the one to watch: the MLP has never seen a "
        "lesion in training."
    )

    # --- figures ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(history.train_loss, label="train")
    ax.plot(history.val_loss, label="validation")
    ax.axvline(
        history.best_epoch,
        color="0.5",
        ls=":",
        label=f"best (epoch {history.best_epoch + 1})",
    )
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE, T1 in seconds")
    ax.set_yscale("log")
    ax.set_title("Per-voxel MLP training", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out / "mlp_training.png", dpi=150)
    plt.close(fig)

    index = max(
        range(len(test_set)), key=lambda i: len(test_set.lesions_for(i))
    )
    mask_i = test_set.mask[index]
    panels = [
        ("Ground truth", np.where(mask_i, truth[index], np.nan), "viridis"),
        ("Fit, with B1", np.where(mask_i, corrected[index], np.nan), "viridis"),
        ("MLP", np.where(mask_i, mlp_maps[index], np.nan), "viridis"),
        (
            "MLP error",
            np.where(mask_i, mlp_maps[index] - truth[index], np.nan),
            "RdBu_r",
        ),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for ax, (title, data, cmap_name) in zip(axes, panels):
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad("0.12")
        kw = (
            dict(vmin=-400, vmax=400)
            if cmap_name == "RdBu_r"
            else dict(vmin=0, vmax=4200)
        )
        im = ax.imshow(data, cmap=cmap, **kw)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        for lesion in test_set.lesions_for(index):
            ax.add_patch(
                plt.Circle(
                    (lesion.col, lesion.row),
                    lesion.radius_px + 2.5,
                    fill=False,
                    color="red",
                    lw=0.9,
                )
            )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    fig.suptitle(
        "Per-voxel MLP on a lesioned test phantom. "
        "Circles mark lesions it never saw in training.",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(args.out / "mlp_vs_classical.png", dpi=150)
    plt.close(fig)

    print(f"\nwrote {(args.out / 'mlp.pt').relative_to(ROOT)}")
    print(f"wrote {(args.out / 'mlp_training.png').relative_to(ROOT)}")
    print(f"wrote {(args.out / 'mlp_vs_classical.png').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
