"""Tests for dataset construction.

The tests that matter here are the ones about leakage and about where
lesions are allowed to appear. Both failures are silent: the code runs, the
figures look good, and the study is worthless. Nothing else in the project
can catch them, so they are pinned here.
"""

import numpy as np
import pytest

from qmri.dataset import (
    Dataset,
    build_splits,
    generate_split,
    normalise_per_voxel,
)

SHAPE = (48, 48)


def small_splits(**kwargs):
    return build_splits(
        n_train=3, n_val=2, n_test=2, shape=SHAPE, seed=0, **kwargs
    )


# --- the two that must never break ----------------------------------------


def test_lesions_appear_only_in_the_test_split():
    """Training on lesions would answer a different, much easier question."""
    splits = small_splits()
    assert len(splits["train"].lesions) == 0
    assert len(splits["val"].lesions) == 0
    assert len(splits["test"].lesions) > 0
    # and the label map agrees with the records
    assert not np.any(splits["train"].labels == 4)
    assert not np.any(splits["val"].labels == 4)
    assert np.any(splits["test"].labels == 4)


def test_splits_share_no_phantom():
    """One phantom is one subject. Anatomy must not cross the split line.

    A slice-level or voxel-level split would let near-identical anatomy sit
    on both sides, and every reported error would come out flatteringly low.
    """
    splits = build_splits(n_train=4, n_val=3, n_test=3, shape=SHAPE, seed=0)

    def fingerprints(ds):
        # Anatomy alone, ignoring lesions, so the test split is comparable.
        return {
            np.asarray(labels == 1).tobytes() for labels in ds.labels
        }

    train, val, test = (fingerprints(splits[k]) for k in ("train", "val", "test"))
    assert not (train & val)
    assert not (train & test)
    assert not (val & test)


# --- reproducibility -------------------------------------------------------


def test_same_seed_gives_identical_splits():
    a = small_splits()
    b = small_splits()
    np.testing.assert_array_equal(a["train"].images, b["train"].images)
    np.testing.assert_array_equal(a["test"].labels, b["test"].labels)


def test_different_seed_gives_different_phantoms():
    a = build_splits(n_train=2, n_val=1, n_test=1, shape=SHAPE, seed=0)
    b = build_splits(n_train=2, n_val=1, n_test=1, shape=SHAPE, seed=1)
    assert not np.array_equal(a["train"].labels, b["train"].labels)


def test_phantom_i_does_not_depend_on_how_many_were_generated():
    # So a split can be extended later without invalidating earlier runs.
    ss = np.random.SeedSequence(7)
    few = generate_split(2, SHAPE, ss)
    ss = np.random.SeedSequence(7)
    many = generate_split(5, SHAPE, ss)
    np.testing.assert_array_equal(few.images[1], many.images[1])


def test_each_phantom_has_its_own_anatomy():
    ds = generate_split(4, SHAPE, np.random.SeedSequence(0))
    unique = {labels.tobytes() for labels in ds.labels}
    assert len(unique) == 4


# --- shapes and contents ---------------------------------------------------


def test_shapes_are_consistent():
    ds = generate_split(3, SHAPE, np.random.SeedSequence(0))
    assert ds.images.shape == (3, 5, *SHAPE)
    assert ds.b1.shape == (3, *SHAPE)
    assert ds.t1.shape == (3, *SHAPE)
    assert ds.labels.shape == (3, *SHAPE)
    assert len(ds) == 3
    assert ds.shape == SHAPE


def test_truth_matches_the_images_that_were_simulated():
    # Background must be dark and tissue bright, i.e. the truth and the
    # images describe the same phantom rather than two different ones.
    ds = generate_split(2, SHAPE, np.random.SeedSequence(0))
    background = ~ds.mask
    assert ds.images[:, 2][background].max() < ds.images[:, 2][ds.mask].mean()
    assert np.all(ds.t1[background] == 0.0)
    assert np.all(ds.t1[ds.mask] > 0.0)


def test_noise_is_actually_applied():
    clean = generate_split(1, SHAPE, np.random.SeedSequence(0), snr=1e9)
    noisy = generate_split(1, SHAPE, np.random.SeedSequence(0), snr=20.0)
    assert not np.allclose(clean.images, noisy.images)
    # same underlying phantom, so the truth is untouched
    np.testing.assert_array_equal(clean.t1, noisy.t1)


def test_b1_amplitude_is_respected():
    ds = generate_split(2, SHAPE, np.random.SeedSequence(0), b1_amplitude=0.3)
    assert np.abs(ds.b1 - 1.0).max() == pytest.approx(0.3, abs=1e-5)


def test_lesions_for_returns_only_that_phantoms_lesions():
    ds = generate_split(
        3, SHAPE, np.random.SeedSequence(0), with_lesions=True, n_lesions=3
    )
    total = sum(len(ds.lesions_for(i)) for i in range(3))
    assert total == len(ds.lesions)
    for i in range(3):
        for lesion in ds.lesions_for(i):
            assert ds.labels[i, lesion.row, lesion.col] == 4


def test_generate_split_rejects_empty():
    with pytest.raises(ValueError, match="at least 1"):
        generate_split(0, SHAPE)


# --- normalisation ---------------------------------------------------------


def test_normalise_removes_scale():
    """Two voxels with the same T1 but different S0 must look identical.

    Otherwise a network can shortcut: recognise the tissue from its
    brightness, then recall that tissue's T1. That is memorising anatomy,
    and it collapses on a lesion whose brightness it has never seen.
    """
    ds = generate_split(1, SHAPE, np.random.SeedSequence(0), snr=1e9)
    scaled = ds.images * 3.7
    np.testing.assert_allclose(
        normalise_per_voxel(ds.images),
        normalise_per_voxel(scaled),
        atol=1e-6,
    )


def test_normalised_voxels_are_unit_length():
    ds = generate_split(1, SHAPE, np.random.SeedSequence(0))
    out = normalise_per_voxel(ds.images)
    norms = np.linalg.norm(out, axis=1)
    tissue = ds.mask
    np.testing.assert_allclose(norms[tissue], 1.0, atol=1e-5)


def test_normalise_leaves_background_finite():
    # Background is exactly zero, so a naive division would give NaN and
    # poison every batch that contains air.
    images = np.zeros((1, 5, 4, 4), dtype=np.float32)
    out = normalise_per_voxel(images)
    assert np.all(np.isfinite(out))
    assert np.all(out == 0.0)


# --- persistence -----------------------------------------------------------


def test_round_trips_through_disk(tmp_path):
    ds = generate_split(
        2, SHAPE, np.random.SeedSequence(0), with_lesions=True, n_lesions=2
    )
    loaded = Dataset.load(ds.save(tmp_path / "x.npz"))

    np.testing.assert_array_equal(ds.images, loaded.images)
    np.testing.assert_array_equal(ds.t1, loaded.t1)
    np.testing.assert_array_equal(ds.labels, loaded.labels)
    assert loaded.tr == ds.tr
    assert loaded.snr == ds.snr
    assert len(loaded.lesions) == len(ds.lesions)
    assert loaded.lesions_for(0) == ds.lesions_for(0)


def test_summary_mentions_the_essentials():
    ds = generate_split(2, SHAPE, np.random.SeedSequence(0))
    text = ds.summary()
    assert "2 phantoms" in text
    assert "48x48" in text
