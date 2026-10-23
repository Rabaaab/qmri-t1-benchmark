"""Did the lesion survive?

A lesion is only visible because it *differs from its surroundings*. So the
thing to measure is not the T1 inside it but the **contrast** across its
edge, and how much of that contrast each method gives back.

    true contrast      = true T1 in lesion  -  true T1 in surrounding tissue
    estimated contrast = same subtraction, using the method's estimates
    retention          = estimated / true

A retention of 1.0 means the lesion came through intact. 0.0 means it was
erased and the method simply painted white matter over it.

**Why not just average the error inside the lesion?** Because that cannot
tell "erased" from "shifted". A method reading 25 ms high *everywhere*
reads 25 ms high inside the lesion too, but the lesion is still perfectly
visible -- everything moved together. Subtracting the surroundings cancels
any such global bias, which is exactly what we want: a global offset is a
different failure from erasure and should not be scored as one.

Three choices are baked in here. They are defensible, not inevitable, and
each is the kind of thing an examiner may ask about.

**Mean, not peak.** ``ratio`` averages over the lesion. For a 4-voxel lesion
that is harsh, since averaging is itself a kind of smoothing -- so
``peak_ratio`` is reported alongside, using the most extreme voxel instead.
Mean is the primary number because a single noisy voxel can dominate a peak.

**A gap between lesion and reference ring.** If a method smeared the lesion
outward, that smear lands in the ring and contaminates the very reference we
are measuring against. Skipping ``gap_px`` pixels before starting the ring
keeps the reference clean. Too small and it is contaminated; too large and
it samples tissue that may genuinely differ. 2 px is a compromise, and
:func:`retention_sensitivity` exists so the choice can be shown not to drive
the conclusion.

**No minimum lesion size.** Tiny lesions are the whole point, so excluding
them would discard the result. Instead every lesion is reported with its
size and the number of voxels behind it, and the analysis bins by size. A
lesion is skipped only when it is *impossible* to score -- no surrounding
host tissue to compare against, or a true contrast so near zero that the
ratio is meaningless.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qmri.phantom import Lesion, disc_mask

__all__ = [
    "Retention",
    "lesion_masks",
    "lesion_retention",
    "retention_sensitivity",
]

DEFAULT_GAP_PX = 2.0
DEFAULT_RING_PX = 3.0

# Below this the lesion is not really a lesion and the ratio explodes.
MIN_TRUE_CONTRAST_MS = 20.0
# A ring smaller than this gives a noisy reference.
MIN_RING_VOXELS = 8


@dataclass(frozen=True)
class Retention:
    """How much of one lesion's contrast a method gave back."""

    lesion: Lesion
    true_contrast: float
    estimated_contrast: float
    ratio: float
    peak_ratio: float
    n_core: int
    n_ring: int

    @property
    def diameter_px(self) -> float:
        """Lesion size as a reader would describe it: pixels across."""
        return 2.0 * self.lesion.radius_px


def lesion_masks(
    shape: tuple[int, int],
    lesion: Lesion,
    gap_px: float = DEFAULT_GAP_PX,
    ring_px: float = DEFAULT_RING_PX,
) -> tuple[np.ndarray, np.ndarray]:
    """The lesion itself, and the reference ring around it.

    Returns ``(core, ring)`` boolean masks. The ring starts ``gap_px``
    outside the lesion edge and is ``ring_px`` wide; the gap exists so that
    a method which smeared the lesion outward does not contaminate its own
    reference.
    """
    if gap_px < 0 or ring_px <= 0:
        raise ValueError(
            f"need gap_px >= 0 and ring_px > 0, got {gap_px} and {ring_px}"
        )
    centre = (lesion.row, lesion.col)
    core = disc_mask(shape, centre, lesion.radius_px)
    inner = disc_mask(shape, centre, lesion.radius_px + gap_px)
    outer = disc_mask(shape, centre, lesion.radius_px + gap_px + ring_px)
    return core, outer & ~inner


def lesion_retention(
    estimate: np.ndarray,
    truth: np.ndarray,
    lesion: Lesion,
    host_mask: np.ndarray,
    gap_px: float = DEFAULT_GAP_PX,
    ring_px: float = DEFAULT_RING_PX,
) -> Retention | None:
    """Score one lesion in one image.

    Parameters
    ----------
    estimate : ndarray, shape (H, W)
        A method's T1 map in milliseconds. NaNs are ignored.
    truth : ndarray, shape (H, W)
        The ground-truth T1 map.
    lesion : Lesion
        The record of what was inserted.
    host_mask : ndarray of bool, shape (H, W)
        Tissue the reference ring may be drawn from -- normally the host
        tissue *excluding* every lesion, so one lesion is never used as the
        reference for another.
    gap_px, ring_px : float
        Reference ring geometry. See the module docstring.

    Returns
    -------
    Retention, or None when the lesion cannot be scored at all.
    """
    core, ring = lesion_masks(estimate.shape, lesion, gap_px, ring_px)
    ring = ring & np.asarray(host_mask, dtype=bool)

    if not core.any() or int(ring.sum()) < MIN_RING_VOXELS:
        return None

    true_core = float(np.nanmean(truth[core]))
    true_ring = float(np.nanmean(truth[ring]))
    true_contrast = true_core - true_ring
    if abs(true_contrast) < MIN_TRUE_CONTRAST_MS:
        return None

    est_core_values = estimate[core][np.isfinite(estimate[core])]
    est_ring_values = estimate[ring][np.isfinite(estimate[ring])]
    if est_core_values.size == 0 or est_ring_values.size == 0:
        return None

    est_ring = float(est_ring_values.mean())
    est_contrast = float(est_core_values.mean()) - est_ring

    # Peak: the most extreme voxel in the direction the lesion actually
    # goes, so a bright lesion is scored on its brightest voxel.
    if true_contrast > 0:
        peak = float(est_core_values.max())
        true_peak = float(np.nanmax(truth[core]))
    else:
        peak = float(est_core_values.min())
        true_peak = float(np.nanmin(truth[core]))
    true_peak_contrast = true_peak - true_ring
    peak_ratio = (
        (peak - est_ring) / true_peak_contrast
        if abs(true_peak_contrast) >= MIN_TRUE_CONTRAST_MS
        else float("nan")
    )

    return Retention(
        lesion=lesion,
        true_contrast=true_contrast,
        estimated_contrast=est_contrast,
        ratio=est_contrast / true_contrast,
        peak_ratio=peak_ratio,
        n_core=int(core.sum()),
        n_ring=int(ring.sum()),
    )


def retention_sensitivity(
    estimate: np.ndarray,
    truth: np.ndarray,
    lesion: Lesion,
    host_mask: np.ndarray,
    gaps=(1.0, 2.0, 3.0, 4.0),
    ring_px: float = DEFAULT_RING_PX,
) -> dict[float, float]:
    """Retention as a function of the gap, to show the choice is not driving it.

    The gap width is an arbitrary decision. If the conclusion changed when
    it moved from 2 px to 3 px, the conclusion would be about the gap rather
    than about the method. Run this and report the spread.
    """
    out: dict[float, float] = {}
    for gap in gaps:
        scored = lesion_retention(
            estimate, truth, lesion, host_mask, gap_px=gap, ring_px=ring_px
        )
        out[float(gap)] = float("nan") if scored is None else scored.ratio
    return out
