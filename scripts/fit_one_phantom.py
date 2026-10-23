"""Phantom in, T1 map out, next to the truth it came from.

Run with:

    python scripts/fit_one_phantom.py

Writes ``outputs/phantom_fit.png`` and prints a per-tissue bias and
variance table.

If the fitted map resembles the ground truth, the simulation loop is sound
end to end and the remaining risk in the project is ordinary engineering.

The run also produces the first real result: the same data, fitted twice,
once pretending the scanner delivered the flip angle it promised and once
using the true transmit field. The difference between those two panels is
the B1 problem, and the error map in the corner shows its defining
characteristic -- the error is smooth and regional, so whole areas of the
image are wrong together rather than independently.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qmri.fit import fit_image
from qmri.forward import rician_noise, sigma_for_snr, smooth_b1_field, spgr_signal
from qmri.phantom import procedural_brain
from qmri.tissues import TISSUES_3T

TR = 15.0  # ms
ANGLES = [2.0, 5.0, 10.0, 15.0, 20.0]  # degrees
SHAPE = (128, 128)
SNR = 100.0  # at the white-matter signal, near its Ernst angle
B1_AMPLITUDE = 0.15  # plausible 3 T transmit inhomogeneity
SEED = 0

OUT = ROOT / "outputs" / "phantom_fit.png"
TISSUE_NAMES = {t.label: t.name for t in TISSUES_3T if t.t1_ms > 0}


def masked(array, mask):
    """Blank the background so it does not dominate the colour scale."""
    return np.where(mask, array, np.nan)


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # --- 1. the answer key -------------------------------------------------
    phantom = procedural_brain(SHAPE, rng=rng)
    mask = phantom.mask

    # --- 2. corrupt it the way a scanner would -----------------------------
    b1 = smooth_b1_field(SHAPE, amplitude=B1_AMPLITUDE)
    clean = spgr_signal(ANGLES, TR, phantom.t1, s0=phantom.pd, b1=b1)

    # Quote SNR against one reference: white matter near its Ernst angle.
    reference = float(np.median(clean[2][phantom.labels == 1]))
    sigma = sigma_for_snr(reference, SNR)
    images = rician_noise(clean, sigma, rng)

    print(f"phantom {SHAPE[0]}x{SHAPE[1]}, {int(mask.sum())} tissue voxels")
    print(f"{len(ANGLES)} flip angles, TR = {TR:.0f} ms, SNR = {SNR:.0f}")
    print(f"B1 spans {b1.min():.3f} to {b1.max():.3f}, mean {b1.mean():.6f}")
    print()

    # --- 3. fit it twice ---------------------------------------------------
    started = time.perf_counter()
    t1_naive, _ = fit_image(images, ANGLES, TR, b1=1.0, mask=mask)
    t1_corrected, _ = fit_image(images, ANGLES, TR, b1=b1, mask=mask)
    elapsed = time.perf_counter() - started
    print(
        f"voxel-wise fitting took {elapsed:.1f} s for "
        f"{2 * int(mask.sum()):,} independent optimisations "
        f"({1e3 * elapsed / (2 * mask.sum()):.2f} ms each)"
    )
    print()

    # --- 4. score against the truth ---------------------------------------
    print(
        f"{'tissue':<14}{'true T1':>9}"
        f"{'bias (no B1)':>14}{'SD':>8}"
        f"{'bias (B1)':>12}{'SD':>8}"
    )
    print("-" * 65)
    for label, name in TISSUE_NAMES.items():
        where = phantom.tissue_mask(label)
        truth = phantom.t1[where]
        row = [f"{name:<14}{truth.mean():>9.0f}"]
        for estimate in (t1_naive, t1_corrected):
            error = estimate[where] - truth
            row.append(f"{np.nanmean(error):>14.1f}{np.nanstd(error):>8.1f}")
        print("".join(row))
    print()
    print("bias and SD in ms. Bias is the systematic offset, SD the scatter;")
    print("they are different failures and only one is fixed by averaging.")

    # --- 5. look at it -----------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.6))
    t1_kw = dict(cmap="viridis", vmin=0.0, vmax=4200.0)

    panels = [
        (axes[0, 0], masked(phantom.t1, mask), "Ground truth T1", t1_kw),
        (
            axes[0, 1],
            images[2],
            f"One measured image ({ANGLES[2]:.0f} deg, SNR {SNR:.0f})",
            dict(cmap="gray"),
        ),
        (
            axes[0, 2],
            masked(b1, mask),
            f"True B1 field ({b1.min():.2f} to {b1.max():.2f})",
            dict(cmap="RdBu_r", vmin=1 - B1_AMPLITUDE, vmax=1 + B1_AMPLITUDE),
        ),
        (
            axes[1, 0],
            masked(t1_naive, mask),
            "Fitted T1, B1 ignored",
            t1_kw,
        ),
        (
            axes[1, 1],
            masked(t1_corrected, mask),
            "Fitted T1, B1 corrected",
            t1_kw,
        ),
        (
            axes[1, 2],
            masked(t1_naive - phantom.t1, mask),
            "Error when B1 is ignored",
            dict(cmap="RdBu_r", vmin=-400.0, vmax=400.0),
        ),
    ]

    for ax, data, title, kw in panels:
        kw = dict(kw)
        cmap = plt.get_cmap(kw.pop("cmap")).copy()
        cmap.set_bad("0.12")
        im = ax.imshow(data, cmap=cmap, **kw)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    fig.suptitle(
        "The same five images, fitted twice. "
        "The B1 error is smooth and regional, not random.",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    plt.close(fig)

    print()
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
