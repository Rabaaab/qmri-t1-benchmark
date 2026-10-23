"""Tests for the lesion retention metric.

The metric is the headline measurement of the project, so its behaviour is
pinned on cases where the right answer is known by construction: a perfect
estimate must score 1, a fully erased lesion must score 0, and -- the one
that matters most -- a uniform global bias must *not* count as erasure.
"""

import numpy as np
import pytest

from qmri.phantom import Lesion, disc_mask
from qmri.retention import (
    lesion_masks,
    lesion_retention,
    retention_sensitivity,
)

SHAPE = (64, 64)
WM, LESION_T1 = 832.0, 1201.0
LESION = Lesion(row=32, col=32, radius_px=4.0, t1_ms=LESION_T1)


@pytest.fixture
def scene():
    """Flat white matter with one lesion, and a host mask excluding it."""
    core = disc_mask(SHAPE, (LESION.row, LESION.col), LESION.radius_px)
    truth = np.full(SHAPE, WM)
    truth[core] = LESION_T1
    host = np.ones(SHAPE, dtype=bool) & ~core
    return truth, host, core


# --- the behaviour that defines the metric --------------------------------


def test_perfect_estimate_scores_one(scene):
    truth, host, _ = scene
    out = lesion_retention(truth.copy(), truth, LESION, host)
    assert out is not None
    assert out.ratio == pytest.approx(1.0, abs=1e-9)


def test_erased_lesion_scores_zero(scene):
    """The method painted white matter over the lesion."""
    truth, host, _ = scene
    erased = np.full(SHAPE, WM)
    out = lesion_retention(erased, truth, LESION, host)
    assert out.ratio == pytest.approx(0.0, abs=1e-9)


def test_global_bias_is_not_counted_as_erasure(scene):
    """The property the whole metric exists for.

    A method reading 25 ms high everywhere reads 25 ms high inside the
    lesion too, but the lesion is still perfectly visible: everything moved
    together. Averaging the error inside the lesion mask would call this a
    failure. Contrast does not, because the shift cancels in the
    subtraction.
    """
    truth, host, _ = scene
    shifted = truth + 25.0
    out = lesion_retention(shifted, truth, LESION, host)
    assert out.ratio == pytest.approx(1.0, abs=1e-9)


def test_halved_contrast_scores_one_half(scene):
    truth, host, _ = scene
    faded = WM + 0.5 * (truth - WM)
    out = lesion_retention(faded, truth, LESION, host)
    assert out.ratio == pytest.approx(0.5, abs=1e-9)


def test_exaggerated_contrast_scores_above_one(scene):
    # Over-sharpening is a real failure too, and must be visible as > 1
    # rather than silently clipped.
    truth, host, _ = scene
    boosted = WM + 1.4 * (truth - WM)
    out = lesion_retention(boosted, truth, LESION, host)
    assert out.ratio == pytest.approx(1.4, abs=1e-9)


def test_smoothing_reduces_retention(scene):
    """A blur, which is exactly how a CNN would erase a lesion."""
    from scipy import ndimage

    truth, host, _ = scene
    blurred = ndimage.gaussian_filter(truth, sigma=2.0)
    out = lesion_retention(blurred, truth, LESION, host)
    assert 0.0 < out.ratio < 1.0


# --- geometry --------------------------------------------------------------


def test_ring_does_not_touch_the_lesion(scene):
    _, _, core = scene
    _, ring = lesion_masks(SHAPE, LESION, gap_px=2.0, ring_px=3.0)
    assert not (ring & core).any()


def test_gap_actually_separates_them():
    # Every ring voxel must sit at least the gap away from the lesion edge,
    # otherwise a smeared lesion contaminates its own reference.
    _, ring = lesion_masks(SHAPE, LESION, gap_px=2.0, ring_px=3.0)
    rows, cols = np.nonzero(ring)
    distance = np.hypot(rows - LESION.row, cols - LESION.col)
    assert distance.min() > LESION.radius_px + 1.0


def test_larger_gap_pushes_the_ring_outward():
    _, near = lesion_masks(SHAPE, LESION, gap_px=1.0)
    _, far = lesion_masks(SHAPE, LESION, gap_px=5.0)
    assert not (near & far).any()


def test_ring_is_restricted_to_the_host_mask(scene):
    """One lesion must never serve as the reference for another."""
    truth, host, _ = scene
    # Put a second lesion inside the first one's ring and exclude it.
    intruder = disc_mask(SHAPE, (32, 40), 2.0)
    truth = truth.copy()
    truth[intruder] = LESION_T1
    out_contaminated = lesion_retention(truth, truth, LESION, host)
    out_clean = lesion_retention(truth, truth, LESION, host & ~intruder)
    # With the intruder excluded the reference is pure white matter, so the
    # true contrast is the full 369 ms.
    assert out_clean.true_contrast == pytest.approx(LESION_T1 - WM, abs=1e-9)
    assert out_contaminated.true_contrast < out_clean.true_contrast


def test_rejects_bad_geometry():
    with pytest.raises(ValueError, match="ring_px > 0"):
        lesion_masks(SHAPE, LESION, gap_px=2.0, ring_px=0.0)


# --- cases that cannot be scored ------------------------------------------


def test_returns_none_when_there_is_no_contrast_to_recover(scene):
    """A 'lesion' identical to its surroundings is not a lesion."""
    _, host, _ = scene
    flat = np.full(SHAPE, WM)
    assert lesion_retention(flat, flat, LESION, host) is None


def test_returns_none_when_the_ring_has_no_host_tissue(scene):
    truth, _, _ = scene
    no_host = np.zeros(SHAPE, dtype=bool)
    assert lesion_retention(truth, truth, LESION, no_host) is None


def test_nans_in_the_estimate_are_ignored(scene):
    # The classical fit returns NaN where it fails to converge; those
    # voxels must not poison the whole lesion.
    truth, host, core = scene
    patchy = truth.copy()
    rows, cols = np.nonzero(core)
    patchy[rows[0], cols[0]] = np.nan
    out = lesion_retention(patchy, truth, LESION, host)
    assert out is not None
    assert np.isfinite(out.ratio)


# --- reporting -------------------------------------------------------------


def test_reports_voxel_counts_and_diameter(scene):
    truth, host, _ = scene
    out = lesion_retention(truth, truth, LESION, host)
    assert out.n_core > 0
    assert out.n_ring >= 8
    assert out.diameter_px == pytest.approx(8.0)


def test_tiny_lesions_are_scored_rather_than_dropped():
    """Small lesions are the point, so they must not be silently excluded."""
    tiny = Lesion(row=32, col=32, radius_px=1.0, t1_ms=LESION_T1)
    core = disc_mask(SHAPE, (32, 32), 1.0)
    truth = np.full(SHAPE, WM)
    truth[core] = LESION_T1
    out = lesion_retention(truth, truth, tiny, np.ones(SHAPE, bool) & ~core)
    assert out is not None
    assert out.ratio == pytest.approx(1.0, abs=1e-9)


def test_sensitivity_sweeps_the_gap(scene):
    truth, host, _ = scene
    swept = retention_sensitivity(truth, truth, LESION, host)
    assert set(swept) == {1.0, 2.0, 3.0, 4.0}
    # A perfect estimate scores 1 whatever the gap, so the arbitrary choice
    # provably is not driving the answer.
    for value in swept.values():
        assert value == pytest.approx(1.0, abs=1e-9)
