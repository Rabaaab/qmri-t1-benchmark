"""Tests for the digital phantom.

The phantom is the answer key. A silent error here -- a tissue that never
gets painted, a T1 that does not match its label -- would not crash anything;
it would just make every result in the study wrong in a way that still looks
like a brain.
"""

import numpy as np
import pytest

from qmri.phantom import Phantom, procedural_brain
from qmri.tissues import TISSUES_3T

BY_LABEL = {t.label: t for t in TISSUES_3T}


def test_contains_all_three_tissues():
    # A phantom missing CSF would hide the project's most interesting
    # failure mode, since long-T1 tissue is where bias is worst.
    ph = procedural_brain((128, 128), rng=np.random.default_rng(0))
    present = set(np.unique(ph.labels))
    assert {0, 1, 2, 3} <= present


def test_every_tissue_covers_a_usable_area():
    # Not just present, but big enough for per-tissue statistics.
    ph = procedural_brain((128, 128), rng=np.random.default_rng(0))
    for label in (1, 2, 3):
        assert np.count_nonzero(ph.labels == label) > 100


def test_background_has_zero_t1_and_pd():
    # Zero T1 is the sentinel the forward model turns into exactly zero
    # signal, so the background must be exactly zero, not merely small.
    ph = procedural_brain((64, 64), rng=np.random.default_rng(1))
    background = ph.labels == 0
    assert np.all(ph.t1[background] == 0.0)
    assert np.all(ph.pd[background] == 0.0)


def test_tissue_t1_matches_its_label():
    # Each tissue's voxels must scatter around that tissue's literature
    # value and no other.
    ph = procedural_brain((160, 160), rng=np.random.default_rng(2))
    for label in (1, 2, 3):
        tissue = BY_LABEL[label]
        values = ph.t1[ph.labels == label]
        assert values.mean() == pytest.approx(tissue.t1_ms, rel=0.02)


def test_within_tissue_variation_is_present():
    # A perfectly uniform phantom would flatter any method that smooths:
    # a network could score well by memorising one constant per tissue
    # without estimating anything.
    ph = procedural_brain((160, 160), rng=np.random.default_rng(3))
    white = ph.t1[ph.labels == 1]
    assert white.std() > 0.5 * BY_LABEL[1].t1_spread_ms


def test_t1_is_always_physical():
    ph = procedural_brain((128, 128), rng=np.random.default_rng(4))
    assert np.all(ph.t1 >= 0.0)
    assert np.all(np.isfinite(ph.t1))


def test_mask_is_exactly_the_non_background():
    ph = procedural_brain((64, 64), rng=np.random.default_rng(5))
    np.testing.assert_array_equal(ph.mask, ph.labels > 0)
    # The head should fill a sensible fraction of the field of view.
    coverage = ph.mask.mean()
    assert 0.3 < coverage < 0.8


def test_white_matter_is_interior_to_grey_matter():
    # Anatomy sanity: the cortex wraps the white matter, so white matter
    # voxels must sit closer to the centre on average.
    ph = procedural_brain((128, 128), rng=np.random.default_rng(6))
    rows, cols = np.indices(ph.labels.shape)
    centre = np.array(ph.labels.shape) / 2.0
    radius = np.hypot(rows - centre[0], cols - centre[1])
    assert radius[ph.labels == 1].mean() < radius[ph.labels == 2].mean()


def test_ventricles_put_csf_inside_the_white_matter():
    # CSF appears both as the outer rim and as the ventricles. If the
    # ventricles were lost, long-T1 tissue would only ever sit at the edge
    # of the head, where a network would learn to ignore it.
    ph = procedural_brain((160, 160), rng=np.random.default_rng(7))
    rows, cols = np.indices(ph.labels.shape)
    centre = np.array(ph.labels.shape) / 2.0
    radius = np.hypot(rows - centre[0], cols - centre[1])
    csf_radius = radius[ph.labels == 3]
    inner_csf = csf_radius < radius[ph.labels == 1].mean()
    assert np.count_nonzero(inner_csf) > 50


def test_shape_is_respected_and_anatomy_rescales():
    small = procedural_brain((64, 64), rng=np.random.default_rng(8))
    large = procedural_brain((192, 192), rng=np.random.default_rng(8))
    assert small.labels.shape == (64, 64)
    assert large.labels.shape == (192, 192)
    # Defined in fractional coordinates, so coverage should barely move.
    assert small.mask.mean() == pytest.approx(large.mask.mean(), abs=0.03)


def test_is_reproducible_from_a_seed():
    a = procedural_brain((96, 96), rng=np.random.default_rng(42))
    b = procedural_brain((96, 96), rng=np.random.default_rng(42))
    np.testing.assert_array_equal(a.labels, b.labels)
    np.testing.assert_allclose(a.t1, b.t1)


def test_jitter_anatomy_changes_the_geometry():
    # Needed so a training set holds many heads rather than one repeated.
    a = procedural_brain(
        (96, 96), rng=np.random.default_rng(0), jitter_anatomy=True
    )
    b = procedural_brain(
        (96, 96), rng=np.random.default_rng(1), jitter_anatomy=True
    )
    assert not np.array_equal(a.labels, b.labels)


def test_fixed_anatomy_is_identical_across_seeds():
    # Only the T1 jitter should differ, so figures stay comparable.
    a = procedural_brain((96, 96), rng=np.random.default_rng(0))
    b = procedural_brain((96, 96), rng=np.random.default_rng(999))
    np.testing.assert_array_equal(a.labels, b.labels)
    assert not np.allclose(a.t1, b.t1)


def test_rejects_tissue_sets_missing_a_label():
    with pytest.raises(ValueError, match="missing label"):
        procedural_brain((32, 32), tissues=TISSUES_3T[:2])


def test_tissue_mask_selects_one_label():
    ph = procedural_brain((64, 64), rng=np.random.default_rng(10))
    np.testing.assert_array_equal(ph.tissue_mask(1), ph.labels == 1)


def test_phantom_is_a_frozen_dataclass():
    # The answer key must not be mutable by accident partway through a run.
    ph = procedural_brain((32, 32), rng=np.random.default_rng(11))
    assert isinstance(ph, Phantom)
    with pytest.raises(Exception):
        ph.t1 = np.zeros((32, 32))
