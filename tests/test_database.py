"""Tests for the results database.

The database exists so that numbers in the report can be traced back to the
code that produced them. These tests check that the trace actually holds:
that a run records its settings, that scores attach to the right run, and
that per-lesion rows survive rather than being averaged away on the way in.
"""

import numpy as np
import pytest

from qmri.database import (
    connect,
    fetch_runs,
    record_run,
    record_summary,
    retention_pivot,
)
from qmri.evaluate import TissueStats
from qmri.phantom import Lesion
from qmri.retention import Retention

TISSUE_NAMES = {1: "white matter", 2: "grey matter", 4: "lesion"}


def fake_retention(diameter, ratio, index=0):
    return Retention(
        lesion=Lesion(row=10, col=10, radius_px=diameter / 2, t1_ms=1201.0),
        true_contrast=369.0,
        estimated_contrast=369.0 * ratio,
        ratio=ratio,
        peak_ratio=ratio,
        n_core=max(int(np.pi * (diameter / 2) ** 2), 1),
        n_ring=40,
    )


def fake_summary(ratios, rmse_whole=70.0):
    return {
        "tissues": {
            1: TissueStats(1, bias=0.3, sd=20.2, n_voxels=5000),
            2: TissueStats(2, bias=0.5, sd=33.5, n_voxels=4000),
        },
        "rmse_whole": rmse_whole,
        "rmse_lesions": 30.8,
        "retention_mean": float(np.mean(ratios)),
        "retention_sd": float(np.std(ratios)),
        "retention_by_size": [],
        "n_lesions": len(ratios),
    }


@pytest.fixture
def db(tmp_path):
    return connect(tmp_path / "results.sqlite")


# --- schema ----------------------------------------------------------------


def test_connect_creates_the_schema(db):
    tables = {
        row["name"]
        for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    assert {"runs", "tissue_stats", "lesion_scores"} <= tables


def test_connect_is_idempotent(tmp_path):
    path = tmp_path / "results.sqlite"
    connect(path).close()
    conn = connect(path)  # must not fail on the second open
    assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0


# --- recording -------------------------------------------------------------


def test_records_a_run_and_its_settings(db):
    run_id = record_run(
        db,
        method="unet",
        dataset="test",
        depth=3,
        n_params=483169,
        receptive_field=92,
        seed=0,
        settings={"epochs": 200, "batch_size": 8},
        train_minutes=11.8,
        best_val_loss=0.00562,
    )
    row = db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    assert row["method"] == "unet"
    assert row["receptive_field"] == 92
    assert '"epochs": 200' in row["settings"]
    assert row["created_at"]


def test_records_the_git_commit_for_traceability(db):
    # The point of the column: six weeks later, which code made this number?
    run_id = record_run(db, method="MLP", dataset="test")
    row = db.execute("SELECT git_commit FROM runs WHERE id=?", (run_id,)).fetchone()
    # None is acceptable outside a git checkout, but not an empty string.
    assert row["git_commit"] is None or len(row["git_commit"]) >= 7


def test_summary_attaches_to_the_right_run(db):
    first = record_run(db, method="MLP", dataset="test")
    second = record_run(db, method="unet", dataset="test")
    record_summary(db, first, fake_summary([1.07], 53.1), TISSUE_NAMES)
    record_summary(db, second, fake_summary([0.62], 75.1), TISSUE_NAMES)

    rows = {r["id"]: r for r in fetch_runs(db)}
    assert rows[first]["rmse_whole"] == pytest.approx(53.1)
    assert rows[second]["rmse_whole"] == pytest.approx(75.1)


def test_tissue_stats_are_stored_per_tissue(db):
    run_id = record_run(db, method="unet", dataset="test")
    record_summary(db, run_id, fake_summary([1.0]), TISSUE_NAMES)
    rows = db.execute(
        "SELECT * FROM tissue_stats WHERE run_id=? ORDER BY label", (run_id,)
    ).fetchall()
    assert [r["tissue"] for r in rows] == ["white matter", "grey matter"]
    assert rows[0]["bias_ms"] == pytest.approx(0.3)


def test_every_lesion_is_stored_individually(db):
    """Not just the average, which is the number that hides the effect.

    The whole question is how retention varies with lesion size. Storing
    only a mean over all sizes would discard exactly the signal the project
    exists to measure.
    """
    lesions = [fake_retention(d, r) for d, r in [(3.0, 0.62), (9.0, 0.99)]]
    run_id = record_run(db, method="unet", dataset="test")
    record_summary(
        db, run_id, fake_summary([0.62, 0.99]), TISSUE_NAMES, lesions
    )

    rows = db.execute(
        "SELECT * FROM lesion_scores WHERE run_id=? ORDER BY diameter_px",
        (run_id,),
    ).fetchall()
    assert len(rows) == 2
    assert rows[0]["diameter_px"] == pytest.approx(3.0)
    assert rows[0]["retention"] == pytest.approx(0.62)
    assert rows[1]["retention"] == pytest.approx(0.99)


def test_recording_twice_replaces_rather_than_duplicates(db):
    # Re-running an experiment must not silently double its lesions.
    lesions = [fake_retention(3.0, 0.62)]
    run_id = record_run(db, method="unet", dataset="test")
    for _ in range(2):
        record_summary(db, run_id, fake_summary([0.62]), TISSUE_NAMES, lesions)
    count = db.execute(
        "SELECT COUNT(*) FROM lesion_scores WHERE run_id=?", (run_id,)
    ).fetchone()[0]
    assert count == 1


# --- querying --------------------------------------------------------------


def test_fetch_runs_filters_by_dataset(db):
    record_run(db, method="unet", dataset="test")
    record_run(db, method="unet", dataset="brainweb")
    assert len(fetch_runs(db)) == 2
    assert len(fetch_runs(db, dataset="brainweb")) == 1


def test_retention_pivot_groups_by_method_and_size(db):
    """The headline table, as one query.

    Reproducing this from loose .npz files would mean reloading every model
    and rescoring every lesion.
    """
    for method, small, large in [("unet", 0.62, 0.99), ("MLP", 1.06, 1.07)]:
        run_id = record_run(db, method=method, dataset="test")
        lesions = [fake_retention(3.0, small), fake_retention(9.0, large)]
        record_summary(
            db, run_id, fake_summary([small, large]), TISSUE_NAMES, lesions
        )

    pivot = {(r["method"], r["size_bin"]): r for r in retention_pivot(db)}
    assert pivot[("unet", "2 to 4")]["mean_retention"] == pytest.approx(0.62)
    assert pivot[("unet", "8 to 12")]["mean_retention"] == pytest.approx(0.99)
    assert pivot[("MLP", "2 to 4")]["mean_retention"] == pytest.approx(1.06)
    assert pivot[("unet", "2 to 4")]["n"] == 1


def test_pivot_ignores_other_datasets(db):
    run_id = record_run(db, method="unet", dataset="brainweb")
    record_summary(
        db, run_id, fake_summary([0.5]), TISSUE_NAMES, [fake_retention(3.0, 0.5)]
    )
    assert retention_pivot(db, dataset="test") == []
    assert len(retention_pivot(db, dataset="brainweb")) == 1
