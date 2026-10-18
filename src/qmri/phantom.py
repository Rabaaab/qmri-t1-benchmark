"""A procedural digital brain phantom.

Nested ellipses, not anatomy. This is deliberate.

BrainWeb gives real labelled anatomy and is worth using eventually, but it is
a download, a format, and a dependency, and the question this project asks
does not need it. What the question needs is a label map where every voxel's
true T1 is known exactly, where tissue boundaries exist, and where regions
are large and smooth enough that a convolutional network has something to
exploit. Ellipses provide all three, generate in milliseconds, and give an
unlimited supply of distinct anatomies for a training set.

Two choices here matter more than the shapes:

*Within-tissue variation.* Each tissue gets its literature T1 plus Gaussian
jitter, rather than one constant value. A phantom of perfectly uniform
regions would flatter any method that smooths -- a network could score well
by learning "output the white-matter constant" without estimating anything.
Variation makes that shortcut cost something.

*A visible ventricle.* CSF has a T1 several times longer than white matter,
so the ventricles are where B1 error and noise bias show up most violently.
A phantom without them would hide the project's most interesting failure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qmri.tissues import TISSUES_3T, Tissue

__all__ = ["Phantom", "procedural_brain"]


@dataclass(frozen=True)
class Phantom:
    """A digital phantom and the ground truth that generated it.

    Attributes
    ----------
    labels : ndarray of int
        Tissue label per voxel; 0 is background.
    t1 : ndarray of float
        True T1 in milliseconds, including within-tissue variation. This is
        the answer key every estimator is scored against.
    pd : ndarray of float
        Relative proton density, used as S0 in the forward model.
    """

    labels: np.ndarray
    t1: np.ndarray
    pd: np.ndarray

    @property
    def mask(self) -> np.ndarray:
        """Voxels containing tissue, i.e. everything worth fitting."""
        return self.labels > 0

    def tissue_mask(self, label: int) -> np.ndarray:
        """Voxels belonging to one tissue, for per-tissue error statistics."""
        return self.labels == label


def _ellipse(shape, centre, radii) -> np.ndarray:
    """Boolean mask of an axis-aligned ellipse, in fractional coordinates.

    ``centre`` and ``radii`` are fractions of the image extent along each
    axis, so the same numbers describe the same anatomy at any resolution.
    """
    total = np.zeros(shape, dtype=float)
    for axis, (n, c, r) in enumerate(zip(shape, centre, radii)):
        coord = (np.linspace(0.0, 1.0, n) - c) / r
        bcast = [1] * len(shape)
        bcast[axis] = -1
        total = total + (coord**2).reshape(bcast)
    return total <= 1.0


def procedural_brain(
    shape: tuple[int, int] = (128, 128),
    rng: np.random.Generator | None = None,
    tissues: tuple[Tissue, ...] = TISSUES_3T,
    jitter_anatomy: bool = False,
) -> Phantom:
    """Build a brain-like phantom with known ground-truth T1.

    Parameters
    ----------
    shape : (int, int)
        Image shape. The anatomy is defined in fractional coordinates, so
        this rescales rather than crops.
    rng : numpy.random.Generator, optional
        Controls within-tissue T1 variation, and the anatomy when
        ``jitter_anatomy`` is set. Pass one explicitly to keep figures
        reproducible.
    tissues : tuple of Tissue
        Must contain labels 1 (white matter), 2 (grey matter) and 3 (CSF).
    jitter_anatomy : bool
        Perturb the ellipse geometry, so that a training set contains many
        distinct heads rather than one repeated. Leave off for a fixed
        reference phantom used in figures.

    Returns
    -------
    Phantom
    """
    if rng is None:
        rng = np.random.default_rng()

    by_label = {t.label: t for t in tissues}
    for required in (1, 2, 3):
        if required not in by_label:
            raise ValueError(f"tissues is missing label {required}")

    def wobble(scale: float) -> float:
        return float(rng.uniform(-scale, scale)) if jitter_anatomy else 0.0

    cy, cx = 0.50 + wobble(0.02), 0.50 + wobble(0.02)

    # Painted outermost first, each layer overwriting the one before.
    head = _ellipse(shape, (cy, cx), (0.46 + wobble(0.03), 0.38 + wobble(0.03)))
    brain = _ellipse(shape, (cy, cx), (0.42 + wobble(0.02), 0.34 + wobble(0.02)))
    white = _ellipse(shape, (cy, cx), (0.33 + wobble(0.03), 0.25 + wobble(0.03)))
    ventricles = _ellipse(
        shape, (cy - 0.06, cx - 0.06), (0.10, 0.045)
    ) | _ellipse(shape, (cy - 0.06, cx + 0.06), (0.10, 0.045))

    labels = np.zeros(shape, dtype=np.int16)
    labels[head] = 3  # CSF rim in the subarachnoid space
    labels[brain] = 2  # grey matter cortex
    labels[white] = 1  # white matter interior
    labels[ventricles & white] = 3  # CSF-filled ventricles

    t1 = np.zeros(shape, dtype=float)
    pd = np.zeros(shape, dtype=float)
    for label, tissue in by_label.items():
        if tissue.t1_ms <= 0:
            continue
        where = labels == label
        n = int(np.count_nonzero(where))
        if n == 0:
            continue
        jitter = rng.normal(0.0, tissue.t1_spread_ms, size=n)
        # Clip rather than allow a non-physical T1; the spreads are small
        # enough relative to the means that this almost never binds.
        t1[where] = np.maximum(tissue.t1_ms + jitter, 1.0)
        pd[where] = tissue.pd

    return Phantom(labels=labels, t1=t1, pd=pd)
