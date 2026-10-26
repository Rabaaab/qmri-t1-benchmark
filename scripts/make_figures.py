"""Regenerate every figure for the report, into a committed folder.

Run with:

    python scripts/make_figures.py

Writes PNGs into ``figures/``, which unlike ``outputs/`` is **tracked by
git**, so the results are visible on the repository page without anyone
having to download data or run a model.

Figures are built from the cached test split, the cached classical fits and
the saved model checkpoints, so this is quick and does not retrain anything.
Any model that has not been trained yet is simply skipped, and the BrainWeb
figure is skipped if that data has not been downloaded.

``retention_vs_size.png`` is the project's headline and the one to show
anybody who asks what this is about.
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

from qmri.dataset import Dataset
from qmri.evaluate import all_lesion_retention, rmse
from qmri.methods import load_methods

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


def figure_inputs(test_set) -> None:
    """What the scanner hands you, and what is computed from it.

    Worth showing first to anyone unfamiliar with the project: the five grey
    pictures are the actual measurement, and the colour map is not measured
    at all. It is solved for, pixel by pixel, from how brightness changes
    across those five.
    """
    index = max(range(len(test_set)), key=lambda i: len(test_set.lesions_for(i)))
    mask = test_set.mask[index]
    angles = test_set.angles

    n = len(angles) + 1
    fig, axes = plt.subplots(1, n, figsize=(2.9 * n, 3.6))
    for ax, angle, image in zip(axes, angles, test_set.images[index]):
        ax.imshow(image, cmap="gray")
        ax.set_title(f"measured\n{angle:.0f}° flip angle", fontsize=9)

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("0.12")
    im = axes[-1].imshow(
        np.where(mask, test_set.t1[index], np.nan), cmap=cmap, vmin=0, vmax=4200
    )
    axes[-1].set_title("computed\nT1 map (ms)", fontsize=9)
    fig.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.03)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        "The five grey images are the measurement. The colour map is solved "
        "for from how brightness changes across them.",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "inputs_and_output.png", dpi=150)
    plt.close(fig)


def figure_retention(methods, test_set) -> None:
    """The headline: how much of a lesion survives, against its size."""
    fig, ax = plt.subplots(figsize=(8, 5))
    edges = np.array([2.0, 4.0, 6.0, 8.0, 12.1])
    centres = (edges[:-1] + edges[1:]) / 2

    ax.axhline(1.0, color="0.3", ls="--", lw=1, zorder=0)
    ax.text(
        centres[-1], 1.005, "lesion fully preserved", ha="right", va="bottom",
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
        x - width / 2, whole, width,
        label="whole image (the number usually quoted)", color="#4c78a8",
    )
    ax.bar(
        x + width / 2, lesion, width,
        label="lesions only (what a doctor needs)", color="#e45756",
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

    fig, axes = plt.subplots(1, len(panels), figsize=(3.1 * len(panels), 3.8))
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
        "T1 maps from the same five images. "
        "Circles mark lesions no model saw in training.",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "t1_maps.png", dpi=150)
    plt.close(fig)


def figure_brainweb() -> None:
    """Real anatomy from BrainWeb, if it has been downloaded.

    Skipped silently when the data is absent, since it is gitignored and a
    fresh clone will not have it.
    """
    from qmri.brainweb import (
        BRAINWEB_TISSUES,
        DEFAULT_PATTERN,
        FuzzyPhantom,
        load_raw_volume,
        mixed_signal,
        purity_mask,
        true_t1_map,
    )

    directory = DATA / "brainweb"
    volumes, tissues = [], []
    for key, tissue in BRAINWEB_TISSUES.items():
        path = directory / DEFAULT_PATTERN.format(key=key)
        if not path.exists():
            return
        volumes.append(load_raw_volume(path))
        tissues.append(tissue)

    stack = np.stack(volumes)
    best = int(np.argmax((stack[3] > 0.5).sum(axis=(1, 2))))
    phantom = FuzzyPhantom(stack[:, best], tuple(tissues))

    images = mixed_signal(phantom, [2.0, 5.0, 10.0, 15.0, 20.0], 15.0)
    brain = phantom.total > 0.5
    pure = purity_mask(phantom, 0.95)
    truth = np.where(brain, true_t1_map(phantom), np.nan)
    lesion = phantom.fractions[3]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("0.12")

    axes[0].imshow(images[2], cmap="gray")
    axes[0].set_title("simulated scan, 10° flip angle", fontsize=9)

    im = axes[1].imshow(truth, cmap=cmap, vmin=0, vmax=4200)
    axes[1].set_title("true T1 (ms)", fontsize=9)
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.03)

    axes[2].imshow(lesion, cmap="magma", vmin=0, vmax=1)
    axes[2].set_title(
        f"MS lesions ({int((lesion > 0.5).sum())} voxels)", fontsize=9
    )

    axes[3].imshow(np.where(brain, pure, np.nan), cmap="RdYlGn", vmin=0, vmax=1)
    axes[3].set_title(
        f"scorable voxels: {100 * pure[brain].mean():.0f}%\n"
        "(red = partial volume, excluded)",
        fontsize=9,
    )

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f"BrainWeb MS phantom, axial slice {best} — real anatomy, "
        "real lesion shapes, simulated signal",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "brainweb_slice.png", dpi=150)
    plt.close(fig)


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    test_path = DATA / "test.npz"
    if not test_path.exists():
        print(f"missing {test_path}; run scripts/build_dataset.py first")
        return 1

    test_set = Dataset.load(test_path)
    methods = load_methods(test_set, DATA, OUTPUTS)
    if not methods:
        print("no methods available; run fit_baseline.py and a training script")
        return 1

    print(f"scoring {len(methods)} methods on {len(test_set)} test phantoms:")
    for name in methods:
        print(f"  - {name}")

    figure_inputs(test_set)
    figure_retention(methods, test_set)
    figure_the_trap(methods, test_set)
    figure_maps(methods, test_set)
    figure_brainweb()

    print()
    for path in sorted(FIGURES.glob("*.png")):
        print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
