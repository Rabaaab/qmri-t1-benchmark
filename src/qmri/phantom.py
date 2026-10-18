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
from scipy import ndimage

from qmri.tissues import LESION_3T, TISSUES_3T, Tissue

__all__ = ["Phantom", "procedural_brain", "Lesion", "insert_lesions"]


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


@dataclass(frozen=True)
class Lesion:
    """One inserted lesion, recorded so its fate can be measured.

    Retention has to be scored per lesion and binned by size, because the
    whole hypothesis is that small ones vanish while large ones survive. A
    single average over all lesions would hide exactly the effect.
    """

    row: int
    col: int
    radius_px: float
    t1_ms: float

    @property
    def area_px(self) -> float:
        return float(np.pi * self.radius_px**2)


def _disc(shape, centre_px, radius_px) -> np.ndarray:
    """Boolean mask of a filled disc, in pixel coordinates."""
    rows, cols = np.ogrid[: shape[0], : shape[1]]
    return (rows - centre_px[0]) ** 2 + (cols - centre_px[1]) ** 2 <= radius_px**2


def insert_lesions(
    phantom: Phantom,
    rng: np.random.Generator,
    n_lesions: int = 4,
    radius_px: tuple[float, float] = (1.0, 6.0),
    host_label: int = 1,
    lesion: Tissue = LESION_3T,
) -> tuple[Phantom, list[Lesion]]:
    """Paste synthetic lesions into the host tissue of an existing phantom.

    Lesions belong in the **test set only**. A network that saw lesions in
    training would learn to preserve them, which is a different and much
    easier question than the one this project asks: what happens to a
    structure the model has never been given a reason to keep.

    Each lesion is placed fully inside the host tissue, far enough from its
    boundary that the surrounding context is uniform. A lesion straddling
    the grey/white boundary would confound retention with edge behaviour.

    Parameters
    ----------
    phantom : Phantom
        The clean phantom to modify. It is not mutated; a new one is
        returned.
    rng : numpy.random.Generator
        Controls count, placement and size.
    n_lesions : int
        How many to attempt. Lesions that cannot be placed without touching
        the host boundary are skipped, so the returned list may be shorter.
    radius_px : (float, float)
        Inclusive range of lesion radii in pixels. The default spans roughly
        2 to 12 pixels across, which brackets the size at which the
        hypothesis says a CNN starts erasing them.
    host_label : int
        Tissue the lesions sit inside. White matter by default.
    lesion : Tissue
        Properties of the lesion tissue, including its label.

    Returns
    -------
    (phantom, lesions)
        A new Phantom with the lesions painted in, and the record of what
        was inserted.
    """
    if n_lesions < 0:
        raise ValueError(f"n_lesions must be non-negative, got {n_lesions}")
    lo, hi = float(radius_px[0]), float(radius_px[1])
    if not 0 < lo <= hi:
        raise ValueError(f"radius_px must be positive and ordered, got {radius_px}")

    labels = phantom.labels.copy()
    t1 = phantom.t1.copy()
    pd = phantom.pd.copy()

    # Shrinks as lesions are placed, so two can never overlap or touch.
    # Overlapping lesions would merge into one larger structure and destroy
    # the per-lesion, per-size retention scoring the project depends on.
    available = phantom.labels == host_label

    placed: list[Lesion] = []
    for _ in range(n_lesions):
        radius = float(rng.uniform(lo, hi))
        # Only centres this far inside the available region leave the lesion
        # fully surrounded by uniform host tissue.
        margin = int(np.ceil(radius)) + 1
        interior = ndimage.binary_erosion(
            available, structure=np.ones((3, 3), dtype=bool), iterations=margin
        )
        candidates = np.flatnonzero(interior)
        if candidates.size == 0:
            continue  # no room left for a lesion this large

        flat = int(rng.choice(candidates))
        row, col = np.unravel_index(flat, labels.shape)
        disc = _disc(labels.shape, (row, col), radius)

        labels[disc] = lesion.label
        n = int(np.count_nonzero(disc))
        jitter = rng.normal(0.0, lesion.t1_spread_ms, size=n)
        t1[disc] = np.maximum(lesion.t1_ms + jitter, 1.0)
        pd[disc] = lesion.pd

        # Claim the lesion plus a one-pixel gap, so the next one cannot
        # touch it.
        available[_disc(labels.shape, (row, col), radius + 1.0)] = False

        placed.append(
            Lesion(
                row=int(row),
                col=int(col),
                radius_px=radius,
                t1_ms=float(t1[disc].mean()),
            )
        )

    return Phantom(labels=labels, t1=t1, pd=pd), placed
