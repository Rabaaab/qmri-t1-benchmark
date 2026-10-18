"""Generate and cache the train / validation / test splits.

Run with:

    python scripts/build_dataset.py

or, for a quick look without waiting:

    python scripts/build_dataset.py --n-train 8 --n-val 4 --n-test 4

Writes ``data/train.npz``, ``data/val.npz``, ``data/test.npz`` and a preview
figure at ``outputs/dataset_preview.png``.

Generating once and caching matters: the classical baseline takes minutes per
image, so every experiment downstream must run against the *same* fixed
data. Everything here is reproducible from ``--seed`` alone, and the files
are gitignored precisely because the simulator, not its output, is the thing
under version control.

Remember what the splits mean. Lesions exist only in the test set, because
the question is what a network does to a structure it was never trained to
keep. Splits are by phantom, never by slice.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qmri.dataset import DEFAULT_SNR, build_splits

PREVIEW = ROOT / "outputs" / "dataset_preview.png"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n-train", type=int, default=120)
    p.add_argument("--n-val", type=int, default=20)
    p.add_argument("--n-test", type=int, default=40)
    p.add_argument("--size", type=int, default=128, help="image side in pixels")
    p.add_argument("--snr", type=float, default=DEFAULT_SNR)
    p.add_argument("--n-lesions", type=int, default=4, help="per test phantom")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=ROOT / "data")
    return p.parse_args(argv)


def preview(test_set, path: Path) -> None:
    """Show one test phantom, its lesions, and what the scanner sees."""
    index = max(range(len(test_set)), key=lambda i: len(test_set.lesions_for(i)))
    lesions = test_set.lesions_for(index)

    truth = np.where(test_set.mask[index], test_set.t1[index], np.nan)
    lesion_mask = test_set.labels[index] == 4

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("0.12")
    im = axes[0].imshow(truth, cmap=cmap, vmin=0, vmax=4200)
    axes[0].set_title(f"True T1, phantom {index}", fontsize=9)
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.03)

    axes[1].imshow(test_set.images[index, 2], cmap="gray")
    axes[1].set_title(
        f"Measured image, {test_set.angles[2]:.0f} deg, SNR {test_set.snr:.0f}",
        fontsize=9,
    )

    axes[2].imshow(lesion_mask, cmap="gray")
    axes[2].set_title(
        f"{len(lesions)} lesions, radii "
        + ", ".join(f"{le.radius_px:.1f}" for le in lesions)
        + " px",
        fontsize=9,
    )

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for le in lesions:
            ax.add_patch(
                plt.Circle(
                    (le.col, le.row),
                    le.radius_px + 2.5,
                    fill=False,
                    color="red",
                    lw=0.9,
                )
            )

    fig.suptitle(
        "Lesions are present in the test split only. "
        "Circles mark them; they are deliberately hard to see.",
        fontsize=10,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    args = parse_args(argv)
    shape = (args.size, args.size)

    print(
        f"generating {args.n_train}/{args.n_val}/{args.n_test} phantoms "
        f"at {args.size}x{args.size}, SNR {args.snr:.0f}, seed {args.seed}"
    )
    started = time.perf_counter()
    splits = build_splits(
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        shape=shape,
        seed=args.seed,
        n_lesions=args.n_lesions,
        snr=args.snr,
    )
    print(f"generated in {time.perf_counter() - started:.1f} s\n")

    for name in ("train", "val", "test"):
        path = splits[name].save(args.out / f"{name}.npz")
        size_mb = path.stat().st_size / 1024**2
        print(f"{name:<6} {splits[name].summary()}")
        print(f"       -> {path.relative_to(ROOT)} ({size_mb:.1f} MB on disk)")

    # Cheap structural checks, in case someone edits build_splits later.
    assert len(splits["train"].lesions) == 0, "lesions leaked into training"
    assert len(splits["val"].lesions) == 0, "lesions leaked into validation"
    assert len(splits["test"].lesions) > 0, "test split has no lesions"

    radii = splits["test"].lesions["radius_px"]
    print()
    print(
        f"test lesions: {len(radii)} total, radii {radii.min():.1f}"
        f" to {radii.max():.1f} px "
        f"({2 * radii.min():.0f} to {2 * radii.max():.0f} px across)"
    )

    preview(splits["test"], PREVIEW)
    print(f"wrote {PREVIEW.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
