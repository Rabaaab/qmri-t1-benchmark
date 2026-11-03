"""Predicted-versus-true scatter: the regression answer to a confusion matrix.

A confusion matrix needs classes. This model predicts a number, so the
equivalent diagnostic is to plot every prediction against its truth. A
perfect method lies exactly on the diagonal; everything else shows you
*where* and *how* it goes wrong, which a single error number cannot.

Lesion voxels are drawn on top in a separate colour, because they are 2% of
the data and would otherwise be invisible under the bulk.
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
from qmri.methods import load_methods


def main() -> int:
    test_set = Dataset.load(ROOT / "data" / "test.npz")
    methods = load_methods(test_set, ROOT / "data", ROOT / "outputs")
    if not methods:
        print("no methods found")
        return 1

    rng = np.random.default_rng(0)
    tissue = test_set.mask & (test_set.labels != 4)
    lesion = test_set.labels == 4

    fig, axes = plt.subplots(1, len(methods), figsize=(4.1 * len(methods), 4.3))
    for ax, (name, estimate) in zip(np.atleast_1d(axes), methods.items()):
        # Thin the bulk: 350k points renders slowly and hides structure.
        y = test_set.t1[tissue]
        p = estimate[tissue]
        keep = rng.choice(y.size, size=min(12000, y.size), replace=False)
        ax.scatter(y[keep], p[keep], s=1, alpha=0.12, color="#4c78a8",
                   label="ordinary tissue", rasterized=True)
        ax.scatter(test_set.t1[lesion], estimate[lesion], s=3, alpha=0.35,
                   color="#e45756", label="lesions", rasterized=True)

        ax.plot([0, 4600], [0, 4600], color="black", lw=1, ls="--",
                label="perfect")
        ax.set_xlim(0, 4600)
        ax.set_ylim(0, 4600)
        ax.set_aspect("equal")
        ax.set_title(name.replace(" (", "\n("), fontsize=9)
        ax.set_xlabel("true T1 (ms)", fontsize=9)
        ax.grid(alpha=0.2)

    np.atleast_1d(axes)[0].set_ylabel("predicted T1 (ms)", fontsize=9)
    np.atleast_1d(axes)[0].legend(fontsize=7.5, loc="upper left",
                                  markerscale=4)
    fig.suptitle(
        "Predicted against true T1. On the dashed line is perfect; "
        "red points are lesions.",
        fontsize=10,
    )
    fig.tight_layout()
    out = ROOT / "figures" / "predicted_vs_true.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("wrote", out.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
