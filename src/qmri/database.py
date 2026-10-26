"""A results database.

Every experiment in this project produces the same shape of answer: one
method, evaluated on one test split, giving per-tissue errors and a score
for every individual lesion. Printing those to a terminal was fine for two
methods. With four methods, several depths, several seeds and two datasets
it stops being fine -- numbers get copied into a report by hand, a run gets
repeated and nobody remembers which version the table came from.

So results go in a SQLite file instead. SQLite is part of Python, needs no
server, and the whole database is a single file you can copy or inspect.

Three tables, in the shape the questions come in:

    runs          one row per (method, dataset) evaluation, with the
                  settings that produced it and the headline scores
    tissue_stats  one row per run per tissue: bias and scatter
    lesion_scores one row per run per lesion: size and retention

Normalised this way, the interesting questions are one query each:

    "retention by lesion size for every architecture"
    "did the deeper model erase more?"
    "which runs used seed 0?"

``git_commit`` is stored on every run. Six weeks from now, when a number in
the report looks wrong, that column says exactly which version of the code
produced it.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path

__all__ = [
    "connect",
    "record_run",
    "record_summary",
    "fetch_runs",
    "retention_pivot",
    "SCHEMA",
]

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT    NOT NULL,
    method          TEXT    NOT NULL,   -- 'fit, with B1', 'MLP', 'unet', 'swin'
    dataset         TEXT    NOT NULL,   -- which split was scored
    depth           INTEGER,            -- NULL for the classical fits
    n_params        INTEGER,
    receptive_field INTEGER,            -- pixels; 1 for voxel-wise methods
    seed            INTEGER,
    settings        TEXT,               -- JSON of the full argument set
    git_commit      TEXT,
    train_minutes   REAL,
    best_val_loss   REAL,
    rmse_whole      REAL,
    rmse_lesions    REAL,
    retention_mean  REAL,
    retention_sd    REAL,
    n_lesions       INTEGER
);

CREATE TABLE IF NOT EXISTS tissue_stats (
    run_id      INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    label       INTEGER NOT NULL,
    tissue      TEXT    NOT NULL,
    bias_ms     REAL    NOT NULL,
    sd_ms       REAL    NOT NULL,
    n_voxels    INTEGER NOT NULL,
    PRIMARY KEY (run_id, label)
);

CREATE TABLE IF NOT EXISTS lesion_scores (
    run_id            INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    lesion_index      INTEGER NOT NULL,
    diameter_px       REAL    NOT NULL,
    true_contrast     REAL    NOT NULL,
    estimated_contrast REAL   NOT NULL,
    retention         REAL    NOT NULL,
    n_core_voxels     INTEGER NOT NULL,
    PRIMARY KEY (run_id, lesion_index)
);

CREATE INDEX IF NOT EXISTS idx_lesion_run ON lesion_scores(run_id);
CREATE INDEX IF NOT EXISTS idx_runs_method ON runs(method);
"""


def _git_commit() -> str | None:
    """The commit the results came from, so a number can be traced back."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return out.stdout.strip() or None
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None


def connect(path: str | Path) -> sqlite3.Connection:
    """Open the database, creating the schema if it is not there yet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def record_run(
    conn: sqlite3.Connection,
    *,
    method: str,
    dataset: str,
    depth: int | None = None,
    n_params: int | None = None,
    receptive_field: int | None = None,
    seed: int | None = None,
    settings: dict | None = None,
    train_minutes: float | None = None,
    best_val_loss: float | None = None,
) -> int:
    """Insert a run and return its id. Scores are attached separately."""
    cursor = conn.execute(
        """INSERT INTO runs (created_at, method, dataset, depth, n_params,
                             receptive_field, seed, settings, git_commit,
                             train_minutes, best_val_loss)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            method,
            dataset,
            depth,
            n_params,
            receptive_field,
            seed,
            json.dumps(settings, default=str) if settings else None,
            _git_commit(),
            train_minutes,
            best_val_loss,
        ),
    )
    conn.commit()
    return int(cursor.lastrowid)


def record_summary(
    conn: sqlite3.Connection,
    run_id: int,
    summary: dict,
    tissue_names: dict[int, str],
    lesions: list | None = None,
) -> None:
    """Attach the scores from :func:`qmri.evaluate.summarise` to a run.

    ``lesions`` is the list of :class:`~qmri.retention.Retention` records.
    Storing them individually rather than only their average is deliberate:
    the whole question is how retention varies with lesion *size*, and an
    average over all sizes is precisely the number that hides it.
    """
    conn.execute(
        """UPDATE runs SET rmse_whole=?, rmse_lesions=?, retention_mean=?,
                           retention_sd=?, n_lesions=? WHERE id=?""",
        (
            summary["rmse_whole"],
            summary["rmse_lesions"],
            summary["retention_mean"],
            summary["retention_sd"],
            summary["n_lesions"],
            run_id,
        ),
    )

    conn.executemany(
        """INSERT OR REPLACE INTO tissue_stats
           (run_id, label, tissue, bias_ms, sd_ms, n_voxels)
           VALUES (?,?,?,?,?,?)""",
        [
            (
                run_id,
                label,
                tissue_names.get(label, f"label {label}"),
                stats.bias,
                stats.sd,
                stats.n_voxels,
            )
            for label, stats in summary["tissues"].items()
        ],
    )

    if lesions:
        conn.executemany(
            """INSERT OR REPLACE INTO lesion_scores
               (run_id, lesion_index, diameter_px, true_contrast,
                estimated_contrast, retention, n_core_voxels)
               VALUES (?,?,?,?,?,?,?)""",
            [
                (
                    run_id,
                    i,
                    r.diameter_px,
                    r.true_contrast,
                    r.estimated_contrast,
                    r.ratio,
                    r.n_core,
                )
                for i, r in enumerate(lesions)
            ],
        )
    conn.commit()


def fetch_runs(conn: sqlite3.Connection, dataset: str | None = None):
    """Every run, newest first, optionally filtered to one dataset."""
    if dataset is None:
        return conn.execute(
            "SELECT * FROM runs ORDER BY id DESC"
        ).fetchall()
    return conn.execute(
        "SELECT * FROM runs WHERE dataset=? ORDER BY id DESC", (dataset,)
    ).fetchall()


def retention_pivot(conn: sqlite3.Connection, dataset: str = "test"):
    """Retention against lesion size, for every method: the headline table.

    Written as one query to show what the database buys. Producing this from
    loose .npz files would mean reloading every model and rescoring every
    lesion.
    """
    return conn.execute(
        """
        SELECT r.method,
               CASE
                 WHEN l.diameter_px <  4 THEN '2 to 4'
                 WHEN l.diameter_px <  6 THEN '4 to 6'
                 WHEN l.diameter_px <  8 THEN '6 to 8'
                 ELSE                          '8 to 12'
               END                    AS size_bin,
               COUNT(*)               AS n,
               ROUND(AVG(l.retention), 3) AS mean_retention
        FROM lesion_scores l
        JOIN runs r ON r.id = l.run_id
        WHERE r.dataset = ?
        GROUP BY r.method, size_bin
        ORDER BY r.method, MIN(l.diameter_px)
        """,
        (dataset,),
    ).fetchall()
