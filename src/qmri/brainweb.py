"""Reading the BrainWeb anatomical phantom.

The procedural phantom in :mod:`qmri.phantom` is nested ellipses. It is
enough to answer the question, but it is not anatomy, and the lesions in it
are discs that someone chose. BrainWeb is a published, citable database from
the Montreal Neurological Institute giving real brain anatomy -- and, in its
MS variants, real lesion shapes in the places lesions actually occur.

Two things change when you move to it, and the second is the important one.

**Real anatomy.** Boundaries are convoluted instead of elliptical, so a
network has genuine structure to learn rather than four smooth rings.

**Partial volume.** BrainWeb does not hand out labels; it hands out *fuzzy
membership* volumes, one per tissue, each voxel holding the fraction of
that tissue present. A voxel on a boundary is genuinely 60% white matter
and 40% CSF.

That matters because you cannot simply average their T1 values. T1 does not
combine linearly -- a voxel half white matter and half CSF does not relax
at the average of the two rates. What combines linearly is the **signal**:
each tissue contributes signal in proportion to how much of the voxel it
occupies. So the forward model runs per tissue and the results are summed
with the fractions as weights, which is what :func:`mixed_signal` does.

It also raises a question with no clean answer: what *is* the true T1 of a
mixed voxel? There isn't one. So scoring is restricted to voxels that are
almost entirely one tissue (see :func:`purity_mask`), and the mixed ones
are excluded and reported as excluded. Including them would mean scoring
every method against a ground truth that does not exist.

Data format: BrainWeb's "raw byte" download is a headerless sequence of
unsigned bytes, X varying fastest and Z slowest, on a 1 mm isotropic grid.
No MINC tools and no NIfTI reader are needed.

Reference: Cocosco CA, Kollokian V, Kwan RK-S, Evans AC. "BrainWeb: Online
Interface to a 3D MRI Simulated Brain Database." NeuroImage 1997;5(4):S425.
Available at https://brainweb.bic.mni.mcgill.ca/
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from qmri.forward import spgr_signal
from qmri.tissues import LESION_3T, TISSUES_3T, Tissue

__all__ = [
    "BRAINWEB_SHAPE",
    "BRAINWEB_TISSUES",
    "load_raw_volume",
    "FuzzyPhantom",
    "load_fuzzy_phantom",
    "mixed_signal",
    "purity_mask",
]

# The standard BrainWeb grid: 181 x 217 x 181 at 1 mm, stored with X
# fastest and Z slowest, so the natural numpy shape is (z, y, x).
BRAINWEB_SHAPE = (181, 217, 181)

# BrainWeb's tissue file names mapped onto the tissues this project models.
# Anything not listed (fat, muscle, skull, skin, vessels) is outside the
# brain and is treated as background.
BRAINWEB_TISSUES: dict[str, Tissue] = {
    "wht": next(t for t in TISSUES_3T if t.label == 1),  # white matter
    "gry": next(t for t in TISSUES_3T if t.label == 2),  # grey matter
    "csf": next(t for t in TISSUES_3T if t.label == 3),  # CSF
    "les": LESION_3T,  # MS lesions, present only in the MS phantoms
}


def load_raw_volume(
    path: str | Path,
    shape: tuple[int, int, int] = BRAINWEB_SHAPE,
    dtype=np.uint8,
) -> np.ndarray:
    """Read one headerless BrainWeb volume as fractions in [0, 1].

    The raw download has no header at all -- the dimensions are printed on
    the download page rather than stored in the file -- so the shape has to
    be supplied. A wrong shape produces a scrambled volume rather than an
    error, so the file size is checked against it.

    Parameters
    ----------
    path : str or Path
        The downloaded ``.rawb`` file for one tissue.
    shape : (int, int, int)
        Volume shape as ``(z, y, x)``.
    dtype
        ``uint8`` for the standard byte download, where 0-255 encodes a
        fraction of 0-1.

    Returns
    -------
    ndarray of float32, shape ``shape``, values in [0, 1].
    """
    path = Path(path)
    data = np.fromfile(path, dtype=dtype)
    expected = int(np.prod(shape))
    if data.size != expected:
        raise ValueError(
            f"{path.name} holds {data.size} voxels but shape {shape} needs "
            f"{expected}. Check the dimensions printed on the download page."
        )
    scale = float(np.iinfo(dtype).max) if np.issubdtype(dtype, np.integer) else 1.0
    return (data.reshape(shape) / scale).astype(np.float32)


@dataclass(frozen=True)
class FuzzyPhantom:
    """Tissue fractions on a slice, plus the tissues they belong to.

    Attributes
    ----------
    fractions : ndarray, shape (n_tissues, H, W)
        How much of each tissue occupies each voxel. Rows correspond to
        ``tissues``. Rows need not sum to 1: the remainder is non-brain
        tissue, which this project treats as background.
    tissues : tuple of Tissue
        One per row of ``fractions``.
    """

    fractions: np.ndarray
    tissues: tuple[Tissue, ...]

    @property
    def total(self) -> np.ndarray:
        """Total brain-tissue fraction per voxel; the brain mask, softly."""
        return self.fractions.sum(axis=0)

    @property
    def labels(self) -> np.ndarray:
        """Dominant tissue per voxel, 0 where there is essentially none.

        For per-tissue statistics only. A voxel's dominant label says
        nothing about how mixed it is, which is what
        :func:`purity_mask` is for.
        """
        out = np.zeros(self.fractions.shape[1:], dtype=np.int16)
        present = self.total > 0.5
        winner = np.argmax(self.fractions, axis=0)
        codes = np.array([t.label for t in self.tissues], dtype=np.int16)
        out[present] = codes[winner[present]]
        return out


def load_fuzzy_phantom(
    directory: str | Path,
    slice_index: int | None = None,
    axis: int = 0,
    pattern: str = "phantom_1.0mm_{key}.rawb",
    shape: tuple[int, int, int] = BRAINWEB_SHAPE,
) -> FuzzyPhantom:
    """Load the tissue volumes in ``directory`` and take one slice.

    Missing tissue files are skipped, so the same call works for the healthy
    phantom (no lesion file) and the MS phantoms (with one).

    Parameters
    ----------
    directory : str or Path
        Where the downloaded ``.rawb`` files live.
    slice_index : int, optional
        Which slice to take. Defaults to the middle of ``axis``, which for
        the default axial orientation lands in the middle of the brain.
    axis : int
        Axis to slice along; 0 is axial on BrainWeb's ``(z, y, x)`` grid.
    pattern : str
        Filename template; ``{key}`` is replaced by the BrainWeb tissue key.
    shape : (int, int, int)
        Volume shape, as for :func:`load_raw_volume`.
    """
    directory = Path(directory)
    rows, tissues = [], []
    for key, tissue in BRAINWEB_TISSUES.items():
        path = directory / pattern.format(key=key)
        if not path.exists():
            continue
        volume = load_raw_volume(path, shape=shape)
        index = volume.shape[axis] // 2 if slice_index is None else slice_index
        rows.append(np.take(volume, index, axis=axis))
        tissues.append(tissue)

    if not rows:
        raise FileNotFoundError(
            f"no BrainWeb tissue files matching {pattern!r} in {directory}. "
            f"Expected keys: {', '.join(BRAINWEB_TISSUES)}"
        )
    return FuzzyPhantom(np.stack(rows), tuple(tissues))


def mixed_signal(
    phantom: FuzzyPhantom,
    flip_angles_deg,
    tr: float,
    b1=1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate images from fractional tissue content.

    Each tissue is pushed through the forward model on its own, and the
    results are summed with the volume fractions as weights:

        S(voxel) = sum over tissues of  fraction * S(T1 of that tissue)

    That is the physically correct way to handle a mixed voxel. Averaging
    the T1 values first and simulating once would be wrong, because the
    signal is not linear in T1 -- a voxel half white matter and half CSF
    does not behave like one with the mean of their relaxation times.

    Parameters
    ----------
    phantom : FuzzyPhantom
    flip_angles_deg : 1-D array_like
    tr : float
        Repetition time, milliseconds.
    b1 : float or ndarray
        Relative transmit field, broadcast over the slice.
    rng : numpy.random.Generator, optional
        If given, each tissue's T1 is jittered by its spread, so the
        phantom is not perfectly uniform within a tissue.

    Returns
    -------
    ndarray, shape ``(n_angles, H, W)``
    """
    angles = np.asarray(flip_angles_deg, dtype=float)
    shape = phantom.fractions.shape[1:]
    total = np.zeros((angles.size, *shape), dtype=np.float64)

    for fraction, tissue in zip(phantom.fractions, phantom.tissues):
        t1 = np.full(shape, tissue.t1_ms, dtype=float)
        if rng is not None and tissue.t1_spread_ms > 0:
            t1 = np.maximum(
                t1 + rng.normal(0.0, tissue.t1_spread_ms, size=shape), 1.0
            )
        signal = spgr_signal(angles, tr, t1, s0=tissue.pd, b1=b1)
        total += signal * fraction

    return total.astype(np.float32)


def purity_mask(phantom: FuzzyPhantom, threshold: float = 0.95) -> np.ndarray:
    """Voxels that are almost entirely one tissue.

    A mixed voxel has no single true T1 -- there is no correct answer to
    score an estimator against -- so scoring is restricted to pure voxels
    and the rest are excluded and reported as excluded. Including them would
    penalise every method against a ground truth that does not exist.

    Parameters
    ----------
    phantom : FuzzyPhantom
    threshold : float
        Minimum fraction of a single tissue. 0.95 keeps voxels that are at
        least 95% one thing.
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")
    return phantom.fractions.max(axis=0) >= threshold


def true_t1_map(phantom: FuzzyPhantom) -> np.ndarray:
    """Ground-truth T1 of the dominant tissue, in milliseconds.

    Meaningful only where :func:`purity_mask` is true; elsewhere the voxel
    is a mixture and this value is a convenient fiction.
    """
    out = np.zeros(phantom.fractions.shape[1:], dtype=np.float32)
    winner = np.argmax(phantom.fractions, axis=0)
    present = phantom.total > 0.5
    values = np.array([t.t1_ms for t in phantom.tissues], dtype=np.float32)
    out[present] = values[winner[present]]
    return out
