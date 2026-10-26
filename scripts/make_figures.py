"""Regenerate every figure for the report, into a committed folder.

Run with:

    python scripts/make_figures.py

Writes PNGs into ``figures/``, which unlike ``outputs/`` is **tracked by
git**, so the results are visible on the repository page without anyone
having to download data or run a model.

Figures are built from the cached test split, the cached classical fits and
the saved model checkpoints, so this is quick and does not retrain anything.
Any model that has not been trained yet is simply skipped.

The first figure is the project's headline and the one to show anybody who
asks what this is about.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from qmri.dataset import (
    Dataset,
    scatter_to_maps,
    to_image_tensors,
    to_voxel_table,
)
from qmri.evaluate import all_lesion_retention, rmse
from qmri.models import UNet, VoxelMLP
from qmri.swin import SwinUNet

FIGURES = ROOT / "figures"
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"

# One colour per method, held constant across every figure so the reader
# only has to learn the legend once.
COLOURS = {
    "Classical fit (no B1)": "#999999",
    "Classical fit (+B1)": "#1f77b4",
    "MLP (no context)": "#2ca02c",
    "U-Net (local context)": "#d62728",
    "Swin (global context)": "#9467bd",
}


def load_methods(test_set: Dataset) -> dict[str, np.ndarray]:
    """Every method we can score, as T1 maps over the test split."""
    methods: dict[str, np.ndarray] = {}

    baseline = DATA / "baseline_test.npz"
    if baseline.exists():
        with np.load(baseline) as f:
            methods["Classical fit (no B1)"] = f["naive"]
            methods["Classical fit (+B1)"] = f["corrected"]

    mlp_path = OUTPUTS / "mlp.pt"
    if mlp_path.exists():
        table_x, _, table_mask = to_voxel_table(test_set)
        ck = torch.load(mlp_path, weights_only=False)
        model = VoxelMLP(
            n_features=table_x.shape[1],
            hidden=ck["config"].get("hidden", 256),
            depth=ck["config"].get("depth", 3),
        )
        model.load_state_dict(ck["state_dict"])
        with torch.no_grad():
            flat = model.predict_ms(torch.as_tensor(table_x)).numpy()
        methods["MLP (no context)"] = scatter_to_maps(flat, table_mask)

    image_x, _, _ = to_image_tensors(test_set)
    for path, label, builder in [
        (OUTPUTS / "unet_d3.pt", "U-Net (local context)", UNet),
        (OUTPUTS / "swin_d2.pt", "Swin (global context)", SwinUNet),
    ]:
        if not path.exists():
            continue
        ck = torch.load(path, weights_only=False)
        cfg = ck["config"]
        if builder is UNet:
            model = UNet(
                in_channels=image_x.shape[1],
                base_channels=cfg["base_channels"],
                depth=cfg["depth"],
            )
        else:
            model = SwinUNet(
                in_channels=image_x.shape[1],
                embed_dim=cfg["embed_dim"],
                depth=cfg["depth"],
                num_heads=cfg["num_heads"],
                window=cfg["window"],
                patch_size=cfg["patch_size"],
            )
        model.load_state_dict(ck["state_dict"])
        model.eval()
        maps = np.empty_like(test_set.t1)
        with torch.no_grad():
            for i in range(len(test_set)):
                batch = torch.as_tensor(image_x[i : i + 1], dtype=torch.float32)
                maps[i] = model.predict_ms(batch).numpy()[0]
        methods[label] = maps

    return methods


def figure_retention(methods, test_set) -> None:
    """The headline: how much of a lesion survives, against its size."""
    fig, ax = plt.subplots(figsize=(8, 5))
    edges = np.array([2.0, 4.0, 6.0, 8.0, 12.1])
    centres = (edges[:-1] + edges[1:]) / 2

    ax.axhline(
        1.0, color="0.3", ls="--", lw=1, zorder=0,
    )
    ax.text(
        11.9, 1.005, "lesion fully preserved", ha="right", va="bottom",
        fontsize=8, color="0.3",
    )

    for name, estimate in methods.items():
        scored = all_lesion_retention(estimate, test_set)
        diameters = np.array([r.diameter_px for r in scored])
        ratios = np.array([r.ratio for r in scored])
        means, errors = [], []
        for low, high in zip(edges[:-1], edges[1:]):
            inside = (diameters >= low) & (diameters < high) & np.isfinite(ratios)
            means.append(ratios[inside].mean() if inside.any() else np.nan)
            errors.append(ratios[inside].std() if inside.any() else np.nan)
        ax.errorbar(
            centres, means, yerr=errors, marker="o", capsize=3, lw=2,
            label=name, color=COLOURS.get(name),
        )

    ax.set_xlabel("lesion size (pixels across)")
    ax.set_ylabel("fraction of lesion contrast recovered")
    ax.set_title(
        "Small lesions survive voxel-wise methods and fade under spatial ones",
        fontsize=11,
    )
    ax.set_xticks(centres)
    ax.set_xticklabels(["2–4", "4–6", "6–8", "8–12"])
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "retention_vs_size.png", dpi=150)
    plt.close(fig)


def figure_the_trap(methods, test_set) -> None:
    """Why the usual summary number hides the failure."""
    names = list(methods)
    whole = [rmse(methods[n], test_set.t1, test_set.mask) for n in names]
    lesions_mask = test_set.labels == 4
    lesion = [rmse(methods[n], test_set.t1, lesions_mask) for n in names]

    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(names))
    width = 0.38
    ax.bar(
        x - width / 2, whole, width, label="whole image (the number usually quoted)",
        color="#4c78a8",
    )
    ax.bar(
        x + width / 2, lesion, width, label="lesions only (what a doctor needs)",
        color="#e45756",
    )
    for i, (a, b) in enumerate(zip(whole, lesion)):
        ax.text(i - width / 2, a + 1, f"{a:.0f}", ha="center", fontsize=8)
        ax.text(i + width / 2, b + 1, f"{b:.0f}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" (", "\n(") for n in names], fontsize=8)
    ax.set_ylabel("error, RMSE in ms (lower is better)")
    ax.set_title(
        "Lesions are 2% of the image, so the overall score hides what happens to them",
        fontsize=11,
    )
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "overall_vs_lesion_error.png", dpi=150)
    plt.close(fig)


def figure_maps(methods, test_set) -> None:
    """The same slice, estimated by each method, with lesions circled."""
    index = max(range(len(test_set)), key=lambda i: len(test_set.lesions_for(i)))
    lesions = test_set.lesions_for(index)
    mask = test_set.mask[index]

    panels = [("Ground truth", test_set.t1[index])]
    panels += [(name, estimate[index]) for name, estimate in methods.items()]

    cols = len(panels)
    fig, axes = plt.subplots(1, cols, figsize=(3.1 * cols, 3.8))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("0.12")

    for ax, (title, data) in zip(np.atleast_1d(axes), panels):
        im = ax.imshow(
            np.where(mask, data, np.nan), cmap=cmap, vmin=0, vmax=4200
        )
        ax.set_title(title.replace(" (", "\n("), fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        for lesion in lesions:
            ax.add_patch(
                plt.Circle(
                    (lesion.col, lesion.row), lesion.radius_px + 3,
                    fill=False, color="red", lw=0.9,
                )
            )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    fig.suptitle(
        "T1 maps from the same five images. Circles mark lesions no model saw in training.",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "t1_maps.png", dpi=150)
    plt.close(fig)


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    test_path = DATA / "test.npz"
    if not test_path.exists():
        print(f"missing {test_path}; run scripts/build_dataset.py first")
        return 1

    test_set = Dataset.load(test_path)
    methods = load_methods(test_set)
    if not methods:
        print("no methods available; run fit_baseline.py and a training script")
        return 1

    print(f"scoring {len(methods)} methods on {len(test_set)} test phantoms:")
    for name in methods:
        print(f"  - {name}")

    figure_retention(methods, test_set)
    figure_the_trap(methods, test_set)
    figure_maps(methods, test_set)

    print()
    for path in sorted(FIGURES.glob("*.png")):
        print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
