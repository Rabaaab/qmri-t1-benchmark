"""Day 1: plot signal against flip angle and check it looks like the physics.

Run with:

    python scripts/day1_signal_curve.py

Writes ``outputs/day1_signal_curve.png`` and prints a noiseless round-trip
check to the console.

What you are looking for in the left panel: each curve rises, peaks at its
Ernst angle, and falls. Short-T1 tissue peaks later and higher. That spread
between the curves is the only reason this method works -- it is the signal
that lets five brightness readings pin down one T1. Where the curves touch,
at very small flip angles, the measurement carries almost no information,
which is why the sampled angles must straddle the peaks.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")  # write a file, never try to open a window
import matplotlib.pyplot as plt
import numpy as np

from qmri.fit import fit_voxel
from qmri.forward import spgr_signal
from qmri.tissues import TISSUES_3T

TR = 15.0  # ms
SAMPLED_ANGLES = [2.0, 5.0, 10.0, 15.0, 20.0]  # degrees
OUT = ROOT / "outputs" / "day1_signal_curve.png"

# White matter, grey matter, CSF -- skip the background label.
TISSUES = [t for t in TISSUES_3T if t.t1_ms > 0]


def ernst_angle_deg(tr: float, t1: float) -> float:
    """The flip angle maximising SPGR signal: cos(alpha_E) = exp(-TR/T1)."""
    return float(np.rad2deg(np.arccos(np.exp(-tr / t1))))


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)

    dense = np.linspace(0.1, 40.0, 800)
    fig, (ax_curve, ax_sampled) = plt.subplots(1, 2, figsize=(11, 4.2))

    for tissue in TISSUES:
        curve = spgr_signal(dense, TR, tissue.t1_ms, s0=tissue.pd).ravel()
        line, = ax_curve.plot(
            dense, curve, label=f"{tissue.name} (T1 = {tissue.t1_ms:.0f} ms)"
        )
        ax_curve.axvline(
            ernst_angle_deg(TR, tissue.t1_ms),
            color=line.get_color(),
            ls=":",
            lw=1,
            alpha=0.6,
        )

        sampled = spgr_signal(
            SAMPLED_ANGLES, TR, tissue.t1_ms, s0=tissue.pd
        ).ravel()
        ax_sampled.plot(
            SAMPLED_ANGLES,
            sampled,
            "o-",
            color=line.get_color(),
            label=tissue.name,
        )

    for angle in SAMPLED_ANGLES:
        ax_curve.axvline(angle, color="0.75", lw=0.8, zorder=0)

    ax_curve.set_title(
        f"SPGR signal vs flip angle (TR = {TR:.0f} ms)\n"
        "dotted = Ernst angle, grey = sampled angles",
        fontsize=10,
    )
    ax_curve.set_xlabel("nominal flip angle (degrees)")
    ax_curve.set_ylabel("signal (arbitrary units)")
    ax_curve.legend(fontsize=8)

    ax_sampled.set_title(
        "What the scanner actually gives you:\nfive numbers per voxel",
        fontsize=10,
    )
    ax_sampled.set_xlabel("nominal flip angle (degrees)")
    ax_sampled.set_ylabel("signal (arbitrary units)")
    ax_sampled.set_xticks(SAMPLED_ANGLES)
    ax_sampled.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    plt.close(fig)

    print(f"wrote {OUT.relative_to(ROOT)}")
    print()
    print(f"{'tissue':<14}{'T1 (ms)':>10}{'Ernst (deg)':>13}{'fit (ms)':>12}{'rel err':>12}")
    print("-" * 61)

    worst = 0.0
    for tissue in TISSUES:
        signals = spgr_signal(SAMPLED_ANGLES, TR, tissue.t1_ms, s0=tissue.pd)
        res = fit_voxel(signals, SAMPLED_ANGLES, TR, b1=1.0)
        rel = abs(res.t1 - tissue.t1_ms) / tissue.t1_ms
        worst = max(worst, rel)
        print(
            f"{tissue.name:<14}{tissue.t1_ms:>10.0f}"
            f"{ernst_angle_deg(TR, tissue.t1_ms):>13.1f}"
            f"{res.t1:>12.4f}{rel:>12.2e}"
        )

    print()
    if worst < 1e-6:
        print(f"PASS  worst relative error {worst:.1e} -- simulator and fit agree.")
        return 0
    print(
        f"FAIL  worst relative error {worst:.1e}. The forward model and the "
        "fit disagree about the physics; fix that before building anything else."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
