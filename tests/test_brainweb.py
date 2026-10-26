"""Tests for the BrainWeb reader.

No BrainWeb files are needed: synthetic volumes in the same raw format
exercise every path. That keeps the suite runnable on a fresh clone, and it
also means these tests check the *logic* -- particularly the partial-volume
physics, which is the part that would be wrong in a way nothing else would
catch.
"""

import numpy as np
import pytest

from qmri.brainweb import (
    BRAINWEB_TISSUES,
    FuzzyPhantom,
    load_fuzzy_phantom,
    load_raw_volume,
    mixed_signal,
    purity_mask,
    true_t1_map,
)
from qmri.forward import spgr_signal
from qmri.tissues import TISSUES_3T

SHAPE = (6, 8, 10)  # small stand-in for the real 181x217x181
ANGLES = [2.0, 5.0, 10.0, 15.0, 20.0]
TR = 15.0
WHITE = next(t for t in TISSUES_3T if t.label == 1)
GREY = next(t for t in TISSUES_3T if t.label == 2)


def write_volume(path, values):
    """Write a float array in [0,1] as BrainWeb-style raw bytes."""
    (np.clip(values, 0, 1) * 255).astype(np.uint8).tofile(path)


@pytest.fixture
def brainweb_dir(tmp_path):
    """A miniature BrainWeb: left half white matter, right half grey."""
    white = np.zeros(SHAPE, dtype=np.float32)
    grey = np.zeros(SHAPE, dtype=np.float32)
    white[:, :, :5] = 1.0
    grey[:, :, 5:] = 1.0
    write_volume(tmp_path / "phantom_1.0mm_wht.rawb", white)
    write_volume(tmp_path / "phantom_1.0mm_gry.rawb", grey)
    return tmp_path


# --- reading raw volumes ---------------------------------------------------


def test_reads_a_raw_volume_as_fractions(tmp_path):
    path = tmp_path / "v.rawb"
    write_volume(path, np.full(SHAPE, 1.0))
    volume = load_raw_volume(path, shape=SHAPE)
    assert volume.shape == SHAPE
    assert volume.dtype == np.float32
    np.testing.assert_allclose(volume, 1.0, atol=1 / 255)


def test_wrong_shape_is_caught_rather_than_scrambling_silently():
    """The raw file has no header, so a wrong shape would not error on its own.

    It would reshape into nonsense and every downstream result would be
    quietly wrong, which is why the voxel count is checked.
    """
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".rawb", delete=False) as f:
        write_volume(f.name, np.zeros(SHAPE))
        with pytest.raises(ValueError, match="needs"):
            load_raw_volume(f.name, shape=(99, 99, 99))


def test_byte_values_map_onto_zero_to_one(tmp_path):
    path = tmp_path / "half.rawb"
    np.full(int(np.prod(SHAPE)), 128, dtype=np.uint8).tofile(path)
    assert load_raw_volume(path, shape=SHAPE).mean() == pytest.approx(
        128 / 255, abs=1e-6
    )


# --- assembling a slice ----------------------------------------------------


def test_loads_the_tissues_that_are_present(brainweb_dir):
    phantom = load_fuzzy_phantom(brainweb_dir, shape=SHAPE)
    assert phantom.fractions.shape == (2, SHAPE[1], SHAPE[2])
    assert [t.label for t in phantom.tissues] == [1, 2]


def test_missing_tissue_files_are_skipped_not_fatal(brainweb_dir):
    # The healthy phantom has no lesion file; the MS phantoms do. One code
    # path has to serve both.
    assert "les" in BRAINWEB_TISSUES
    phantom = load_fuzzy_phantom(brainweb_dir, shape=SHAPE)
    assert all(t.label != 4 for t in phantom.tissues)


def test_empty_directory_gives_a_useful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="no BrainWeb tissue files"):
        load_fuzzy_phantom(tmp_path, shape=SHAPE)


def test_dominant_label_picks_the_majority_tissue(brainweb_dir):
    phantom = load_fuzzy_phantom(brainweb_dir, shape=SHAPE)
    labels = phantom.labels
    assert set(np.unique(labels[:, :5])) == {1}
    assert set(np.unique(labels[:, 5:])) == {2}


def test_background_is_label_zero():
    fractions = np.zeros((2, 4, 4), dtype=np.float32)
    fractions[0, 0, 0] = 1.0
    phantom = FuzzyPhantom(fractions, (WHITE, GREY))
    assert phantom.labels[0, 0] == 1
    assert phantom.labels[2, 2] == 0


# --- the partial-volume physics -------------------------------------------


def test_pure_voxel_matches_the_plain_forward_model():
    """A 100% white-matter voxel must give exactly what the simple model gives."""
    fractions = np.ones((1, 2, 2), dtype=np.float32)
    phantom = FuzzyPhantom(fractions, (WHITE,))
    mixed = mixed_signal(phantom, ANGLES, TR)
    plain = spgr_signal(ANGLES, TR, WHITE.t1_ms, s0=WHITE.pd)
    np.testing.assert_allclose(mixed[:, 0, 0], plain.ravel(), rtol=1e-5)


def test_mixed_voxel_blends_signals_not_t1_values():
    """The physics point of this module.

    A voxel half white matter and half grey must give the average of their
    *signals*. Averaging their T1 values and simulating once gives a
    different, wrong answer, because signal is not linear in T1.
    """
    fractions = np.full((2, 1, 1), 0.5, dtype=np.float32)
    phantom = FuzzyPhantom(fractions, (WHITE, GREY))

    mixed = mixed_signal(phantom, ANGLES, TR)[:, 0, 0]

    correct = 0.5 * spgr_signal(ANGLES, TR, WHITE.t1_ms, s0=WHITE.pd).ravel()
    correct += 0.5 * spgr_signal(ANGLES, TR, GREY.t1_ms, s0=GREY.pd).ravel()
    np.testing.assert_allclose(mixed, correct, rtol=1e-5)

    # And it is genuinely different from the naive T1-averaging approach.
    naive = spgr_signal(
        ANGLES, TR, 0.5 * (WHITE.t1_ms + GREY.t1_ms), s0=0.5 * (WHITE.pd + GREY.pd)
    ).ravel()
    assert not np.allclose(mixed, naive, rtol=1e-3)


def test_empty_voxel_gives_no_signal():
    phantom = FuzzyPhantom(np.zeros((2, 3, 3), dtype=np.float32), (WHITE, GREY))
    assert np.all(mixed_signal(phantom, ANGLES, TR) == 0.0)


def test_signal_scales_with_how_much_tissue_is_present():
    half = FuzzyPhantom(np.full((1, 1, 1), 0.5, np.float32), (WHITE,))
    full = FuzzyPhantom(np.full((1, 1, 1), 1.0, np.float32), (WHITE,))
    np.testing.assert_allclose(
        mixed_signal(half, ANGLES, TR), 0.5 * mixed_signal(full, ANGLES, TR),
        rtol=1e-5,
    )


def test_b1_still_applies_through_the_mixture():
    phantom = FuzzyPhantom(np.ones((1, 1, 1), np.float32), (WHITE,))
    distorted = mixed_signal(phantom, [20.0], TR, b1=0.5)
    honest = mixed_signal(phantom, [10.0], TR, b1=1.0)
    np.testing.assert_allclose(distorted, honest, rtol=1e-5)


# --- what may be scored ----------------------------------------------------


def test_purity_mask_keeps_pure_voxels_and_drops_mixtures():
    """A mixed voxel has no single true T1, so it cannot be scored at all."""
    fractions = np.zeros((2, 1, 3), dtype=np.float32)
    fractions[0, 0, 0] = 1.00  # pure white matter
    fractions[0, 0, 1] = 0.60  # a 60/40 mixture
    fractions[1, 0, 1] = 0.40
    fractions[1, 0, 2] = 0.97  # nearly pure grey matter
    mask = purity_mask(FuzzyPhantom(fractions, (WHITE, GREY)), threshold=0.95)
    assert mask.tolist() == [[True, False, True]]


def test_purity_threshold_is_validated():
    phantom = FuzzyPhantom(np.ones((1, 2, 2), np.float32), (WHITE,))
    with pytest.raises(ValueError, match="threshold"):
        purity_mask(phantom, threshold=0.0)


def test_true_t1_map_uses_the_dominant_tissue():
    fractions = np.zeros((2, 1, 2), dtype=np.float32)
    fractions[0, 0, 0] = 1.0
    fractions[1, 0, 1] = 1.0
    t1 = true_t1_map(FuzzyPhantom(fractions, (WHITE, GREY)))
    assert t1[0, 0] == pytest.approx(WHITE.t1_ms)
    assert t1[0, 1] == pytest.approx(GREY.t1_ms)
