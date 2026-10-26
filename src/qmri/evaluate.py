"""Scoring every method the same way.

One module so that the classical fits, the MLP, the U-Net and anything added
later are judged by identical code. If each experiment wrote its own scoring
you could never be sure a difference between two methods was not a
difference between two scripts.

Three numbers per method, and they answer different questions:

- **bias and scatter per tissue** -- is it systematically wrong, or just
  noisy? Those are different failures and only one is fixed by averaging.
- **whole-image RMSE** -- the headline number everyone quotes, and the one
  that hides the thing this project is about.
- **lesion retention by size** -- whether the small structures survived.

Reporting the second without the third is exactly the mistake under
investigation, so :func:`summarise` always returns both.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qmri.dataset import Dataset
from qmri.retention import Retention, lesion_retention

__all__ = [
    "TissueStats",
    "per_tissue_stats",
    "rmse",
    "all_lesion_retention",
    "retention_by_size",
    "summarise",
]

DEFAULT_SIZE_BINS = ((2.0, 4.0), (4.0, 6.0), (6.0, 8.0), (8.0, 12.1))
WHITE_MATTER_LABEL = 1
LESION_LABEL = 4


@dataclass(frozen=True)
class TissueStats:
    """Bias and scatter for one tissue, in milliseconds."""

    label: int
    bias: float
    sd: float
    n_voxels: int


def per_tissue_stats(
    estimate: np.ndarray, truth: np.ndarray, labels: np.ndarray
) -> dict[int, TissueStats]:
    """Signed mean error and its spread, per tissue label.

    NaNs are dropped, since the classical fit leaves them wherever it failed
    to converge and they would otherwise swallow the whole tissue.
    """
    out: dict[int, TissueStats] = {}
    for label in np.unique(labels):
        label = int(label)
        if label == 0:
            continue
        where = labels == label
        error = (estimate[where] - truth[where]).astype(float)
        error = error[np.isfinite(error)]
        if error.size == 0:
            continue
        out[label] = TissueStats(
            label=label,
            bias=float(error.mean()),
            sd=float(error.std()),
            n_voxels=int(error.size),
        )
    return out


def rmse(estimate: np.ndarray, truth: np.ndarray, where: np.ndarray) -> float:
    """Root mean squared error over a region, combining bias and scatter."""
    error = (estimate[where] - truth[where]).astype(float)
    error = error[np.isfinite(error)]
    if error.size == 0:
        return float("nan")
    return float(np.sqrt((error**2).mean()))


def all_lesion_retention(
    estimate: np.ndarray, dataset: Dataset, **kwargs
) -> list[Retention]:
    """Score every lesion in a split.

    The reference ring is drawn from white matter only. Because lesions
    carry their own label, this automatically prevents one lesion being used
    as the reference for another.
    """
    scored: list[Retention] = []
    for i in range(len(dataset)):
        host = dataset.labels[i] == WHITE_MATTER_LABEL
        for lesion in dataset.lesions_for(i):
            result = lesion_retention(
                estimate[i], dataset.t1[i], lesion, host, **kwargs
            )
            if result is not None:
                scored.append(result)
    return scored


def retention_by_size(
    scored: list[Retention], bins=DEFAULT_SIZE_BINS
) -> list[tuple[str, int, float, float]]:
    """Group retention by lesion diameter.

    Binning is the point. A single average over all lesions would hide the
    effect under investigation, which is that *small* lesions vanish while
    large ones survive.

    Returns rows of ``(label, n, mean, sd)``.
    """
    rows = []
    for low, high in bins:
        values = [
            r.ratio for r in scored if low <= r.diameter_px < high and np.isfinite(r.ratio)
        ]
        label = f"{low:g} to {high:g}"
        if values:
            rows.append((label, len(values), float(np.mean(values)), float(np.std(values))))
        else:
            rows.append((label, 0, float("nan"), float("nan")))
    return rows


def summarise(
    estimate: np.ndarray, dataset: Dataset, **kwargs
) -> dict[str, object]:
    """Everything worth recording about one method on one split.

    Always returns the lesion numbers alongside the whole-image one.
    Quoting overall error without them is the precise mistake this project
    exists to document.
    """
    truth, labels = dataset.t1, dataset.labels
    scored = all_lesion_retention(estimate, dataset, **kwargs)
    lesion_voxels = labels == LESION_LABEL

    retentions = [r.ratio for r in scored if np.isfinite(r.ratio)]
    return {
        "tissues": per_tissue_stats(estimate, truth, labels),
        "rmse_whole": rmse(estimate, truth, dataset.mask),
        "rmse_lesions": rmse(estimate, truth, lesion_voxels)
        if lesion_voxels.any()
        else float("nan"),
        "retention_mean": float(np.mean(retentions)) if retentions else float("nan"),
        "retention_sd": float(np.std(retentions)) if retentions else float("nan"),
        "retention_by_size": retention_by_size(scored),
        "n_lesions": len(scored),
    }
