"""Fit the classical baselines over a whole split and cache the result.

Run with:

    python scripts/fit_baseline.py --split test

Writes ``data/baseline_<split>.npz`` holding two T1 maps per phantom: the
naive fit that assumes the scanner delivered the flip angle it promised, and
the corrected fit that is handed the true transmit field.

This is the slow step in the project, and the reason it is a separate
cached artefact rather than part of every experiment. Each phantom is
roughly 8,800 voxels, each fitted twice, as completely independent
two-parameter optimisations that never look at a neighbour -- about two
minutes of CPU per phantom on one core. Phantoms are independent, so the
work parallelises almost perfectly across them.

That runtime is itself a result worth reporting: a trained network produces
the same map in milliseconds.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from qmri.dataset import Dataset
from qmri.fit import fit_image


def _fit_one(task):
    """Fit one phantom both ways. Top-level so it survives pickling."""
    index, images, b1, mask, angles, tr = task
    naive, _ = fit_image(images, angles, tr, b1=1.0, mask=mask)
    corrected, _ = fit_image(images, angles, tr, b1=b1, mask=mask)
    return index, naive.astype(np.float32), corrected.astype(np.float32)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument(
        "--workers",
        type=int,
        default=0,
        help="processes to use; 0 picks cores-1",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="fit only the first N phantoms, for a quick check",
    )
    p.add_argument("--data", type=Path, default=ROOT / "data")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    import os

    source = args.data / f"{args.split}.npz"
    if not source.exists():
        print(f"missing {source}. Run scripts/build_dataset.py first.")
        return 1

    dataset = Dataset.load(source)
    n = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)

    print(f"{args.split}: {dataset.summary()}")
    print(f"fitting {n} phantoms on {workers} workers")

    mask = dataset.mask
    tasks = [
        (
            i,
            dataset.images[i],
            dataset.b1[i].astype(float),
            mask[i],
            dataset.angles,
            dataset.tr,
        )
        for i in range(n)
    ]

    shape = (n, *dataset.shape)
    naive = np.full(shape, np.nan, dtype=np.float32)
    corrected = np.full(shape, np.nan, dtype=np.float32)

    started = time.perf_counter()
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, a, b in pool.map(_fit_one, tasks):
            naive[index] = a
            corrected[index] = b
            done += 1
            elapsed = time.perf_counter() - started
            remaining = elapsed / done * (n - done)
            print(
                f"  {done}/{n} phantoms  "
                f"{elapsed / 60:.1f} min elapsed, "
                f"~{remaining / 60:.1f} min left",
                flush=True,
            )

    elapsed = time.perf_counter() - started
    out = args.data / f"baseline_{args.split}.npz"
    np.savez_compressed(
        out,
        naive=naive,
        corrected=corrected,
        n_phantoms=np.int32(n),
        seconds=np.float64(elapsed),
    )

    voxels = 2 * int(mask[:n].sum())
    print()
    print(
        f"{elapsed / 60:.1f} min for {voxels:,} independent optimisations "
        f"({1e3 * elapsed * workers / voxels:.2f} ms each, single-core equivalent)"
    )
    print(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size / 1024**2:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
