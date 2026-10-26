"""Score every method and record the results in the database.

Run with:

    python scripts/record_results.py

Writes ``data/results.sqlite`` and prints the headline table straight back
out of it, as a single SQL query.

Until now results lived in terminal output. This puts them somewhere they
can be queried, compared across runs, and traced: every row records the git
commit that produced it, so a number that looks wrong in six weeks can be
tied to the exact version of the code behind it.

Re-running is safe. Each run is a new row, so repeating an experiment adds
to the history rather than overwriting it -- which is what you want when
checking whether a result was a fluke.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from qmri.database import connect, record_run, record_summary, retention_pivot
from qmri.dataset import Dataset
from qmri.evaluate import all_lesion_retention, summarise
from qmri.methods import describe_methods, load_methods
from qmri.tissues import LESION_3T, TISSUES_3T

TISSUE_NAMES = {t.label: t.name for t in TISSUES_3T if t.t1_ms > 0}
TISSUE_NAMES[LESION_3T.label] = LESION_3T.name


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--split", default="test")
    p.add_argument("--data", type=Path, default=ROOT / "data")
    p.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    p.add_argument("--db", type=Path, default=ROOT / "data" / "results.sqlite")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    split_path = args.data / f"{args.split}.npz"
    if not split_path.exists():
        print(f"missing {split_path}; run scripts/build_dataset.py first")
        return 1

    test_set = Dataset.load(split_path)
    methods = load_methods(test_set, args.data, args.outputs)
    if not methods:
        print("nothing to record; run fit_baseline.py and a training script")
        return 1

    provenance = describe_methods(args.outputs)
    conn = connect(args.db)

    print(f"scoring {len(methods)} methods on {len(test_set)} phantoms\n")
    for name, estimate in methods.items():
        summary = summarise(estimate, test_set)
        lesions = all_lesion_retention(estimate, test_set)
        info = provenance.get(name, {})

        run_id = record_run(
            conn,
            method=name,
            dataset=args.split,
            depth=info.get("depth"),
            n_params=info.get("n_params"),
            receptive_field=info.get("receptive_field"),
            seed=info.get("seed"),
            settings=info.get("settings"),
            best_val_loss=info.get("best_val_loss"),
        )
        record_summary(conn, run_id, summary, TISSUE_NAMES, lesions)
        print(
            f"  run {run_id:>3}  {name:<24} "
            f"RMSE {summary['rmse_whole']:>6.1f} whole / "
            f"{summary['rmse_lesions']:>6.1f} lesions, "
            f"retention {summary['retention_mean']:.3f}"
        )

    # Read the headline back out of the database, as one query. This is the
    # point of storing it: no model is reloaded and no lesion is rescored.
    print("\nretention by lesion size, straight from SQL:\n")
    rows = retention_pivot(conn, dataset=args.split)
    bins = ["2 to 4", "4 to 6", "6 to 8", "8 to 12"]
    table: dict[str, dict[str, float]] = {}
    counts: dict[str, int] = {}
    for row in rows:
        table.setdefault(row["method"], {})[row["size_bin"]] = row["mean_retention"]
        counts[row["size_bin"]] = row["n"]

    header = f"{'method':<26}" + "".join(f"{b:>12}" for b in bins)
    print(header)
    print("-" * len(header))
    for method, values in table.items():
        line = f"{method:<26}"
        for b in bins:
            value = values.get(b)
            line += f"{value:>12.3f}" if value is not None else f"{'-':>12}"
        print(line)
    print(f"{'lesions per bin':<26}" + "".join(f"{counts.get(b, 0):>12}" for b in bins))

    size = args.db.stat().st_size / 1024
    print(f"\nwrote {args.db.relative_to(ROOT)} ({size:.0f} KB)")
    print(
        "Query it with any SQLite tool, or: "
        "sqlite3 data/results.sqlite 'SELECT * FROM runs'"
    )
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
