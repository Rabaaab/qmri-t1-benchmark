"""Tests for the forward model.

These check the physics, not the plumbing: that the signal equation behaves
the way the SPGR steady state must, and that Rician noise reproduces the
one-directional bias that makes dark regions dangerous.
"""

import numpy as np
import pytest

from qmri.forward import rician_noise, sigma_for_snr, smooth_b1_field, spgr_signal

ANGLES = [2.0, 5.0, 10.0, 15.0, 20.0]
TR = 15.0


def test_zero_flip_angle_gives_zero_signal():
    # No excitation, no transverse magnetisation, no signal.
    assert spgr_signal([0.0], TR, 832.0) == pytest.approx(0.0, abs=1e-15)


def test_background_returns_exactly_zero():
    # T1 <= 0 is the phantom's air label. It must be exactly zero, not a
    # small number, so masks built from it are unambiguous.
    out = spgr_signal(ANGLES, TR, t1=0.0, s0=1000.0)
    assert np.all(out == 0.0)


def test_signal_scales_linearly_with_s0():
    a = spgr_signal(ANGLES, TR, 832.0, s0=1.0)
    b = spgr_signal(ANGLES, TR, 832.0, s0=7.5)
    np.testing.assert_allclose(b, 7.5 * a, rtol=1e-12)


def test_shorter_t1_gives_more_signal_at_high_flip_angle():
    # Faster recovery means more longitudinal magnetisation available each
    # TR, so the shorter-T1 tissue is brighter. This is the contrast the
    # method exploits, and it only appears well away from tiny flip angles.
    short = spgr_signal([15.0], TR, t1=500.0)
    long = spgr_signal([15.0], TR, t1=2000.0)
    assert short > long


def test_signal_has_an_interior_maximum():
    # The Ernst angle: signal rises, peaks, then falls. A monotonic curve
    # would mean the flip angle carried no information worth sampling.
    angles = np.linspace(0.5, 89.0, 400)
    sig = spgr_signal(angles, TR, t1=832.0).ravel()
    peak = int(np.argmax(sig))
    assert 0 < peak < len(angles) - 1


def test_ernst_angle_matches_closed_form():
    # cos(alpha_E) = exp(-TR/T1) is the textbook result; the simulated curve
    # must peak there.
    t1 = 832.0
    expected = np.rad2deg(np.arccos(np.exp(-TR / t1)))
    angles = np.linspace(0.1, 89.9, 9000)
    sig = spgr_signal(angles, TR, t1=t1).ravel()
    assert angles[int(np.argmax(sig))] == pytest.approx(expected, abs=0.05)


def test_b1_scales_the_effective_flip_angle():
    # Asking for 20 degrees at B1 = 0.5 must be indistinguishable from
    # asking for 10 degrees at B1 = 1.0. This equivalence *is* the B1
    # problem: the scanner cannot tell the two cases apart.
    distorted = spgr_signal([20.0], TR, 832.0, b1=0.5)
    honest = spgr_signal([10.0], TR, 832.0, b1=1.0)
    np.testing.assert_allclose(distorted, honest, rtol=1e-12)


def test_output_shape_puts_flip_angle_on_axis_zero():
    t1_map = np.full((8, 6), 832.0)
    out = spgr_signal(ANGLES, TR, t1_map)
    assert out.shape == (len(ANGLES), 8, 6)


def test_map_and_voxel_agree():
    # A whole-image call must give exactly what a per-voxel call gives, or
    # every vectorised experiment silently disagrees with the reference fit.
    rng = np.random.default_rng(0)
    t1_map = rng.uniform(300.0, 3000.0, size=(5, 4))
    b1_map = rng.uniform(0.8, 1.2, size=(5, 4))
    stack = spgr_signal(ANGLES, TR, t1_map, b1=b1_map)
    for i in range(5):
        for j in range(4):
            one = spgr_signal(ANGLES, TR, t1_map[i, j], b1=b1_map[i, j])
            np.testing.assert_allclose(stack[:, i, j], one, rtol=1e-12)


def test_rejects_non_1d_flip_angles():
    with pytest.raises(ValueError, match="1-D"):
        spgr_signal(np.array([[2.0, 5.0]]), TR, 832.0)


def test_rejects_non_positive_tr():
    with pytest.raises(ValueError, match="positive"):
        spgr_signal(ANGLES, 0.0, 832.0)


# --- noise -----------------------------------------------------------------


def test_zero_sigma_leaves_signal_untouched():
    clean = spgr_signal(ANGLES, TR, 832.0, s0=100.0)
    np.testing.assert_array_equal(
        rician_noise(clean, 0.0, np.random.default_rng(0)), clean
    )


def test_rician_noise_is_never_negative():
    rng = np.random.default_rng(1)
    noisy = rician_noise(np.full(10_000, 0.5), sigma=1.0, rng=rng)
    assert np.all(noisy >= 0.0)


def test_rician_noise_inflates_the_signal_at_low_snr():
    # The claim from the project write-up, made testable. Magnitude noise
    # cannot push a sample below zero, so at low SNR its mean sits *above*
    # the true value. This is bias, not scatter, and least squares assumes
    # it away. Expected mean for nu = sigma = 1 is about 1.46.
    rng = np.random.default_rng(2)
    noisy = rician_noise(np.full(400_000, 1.0), sigma=1.0, rng=rng)
    assert noisy.mean() > 1.3


def test_rician_noise_is_unbiased_at_high_snr():
    # Far above the noise floor the distribution is effectively Gaussian and
    # the bias vanishes, which is why the problem hides until you look at
    # dark tissue.
    rng = np.random.default_rng(3)
    noisy = rician_noise(np.full(400_000, 100.0), sigma=1.0, rng=rng)
    assert noisy.mean() == pytest.approx(100.0, rel=1e-3)


def test_sigma_for_snr_round_trips():
    assert 500.0 / sigma_for_snr(500.0, 50.0) == pytest.approx(50.0)


def test_sigma_for_snr_rejects_non_positive():
    with pytest.raises(ValueError, match="positive"):
        sigma_for_snr(500.0, 0.0)


# --- B1 field --------------------------------------------------------------


def test_b1_field_has_mean_exactly_one():
    # A field whose mean drifted from 1.0 would impose a global scale error
    # on top of the regional variation under study, confounding every
    # experiment that uses it.
    field = smooth_b1_field((64, 64), amplitude=0.2)
    assert field.shape == (64, 64)
    assert field.mean() == pytest.approx(1.0, abs=1e-12)


def test_b1_field_respects_the_requested_amplitude():
    amplitude = 0.2
    field = smooth_b1_field((64, 64), amplitude=amplitude)
    # No voxel deviates by more than the amplitude, and the extreme is
    # actually reached rather than merely bounded.
    assert np.abs(field - 1.0).max() == pytest.approx(amplitude, abs=1e-12)
    # Brighter in the middle, dimmer at the corners, as a transmit field is.
    assert field[32, 32] > 1.0
    assert field[0, 0] < 1.0


def test_b1_field_is_smooth():
    # Neighbouring voxels must differ by far less than the global range.
    # A speckled field would average out across a region; a smooth one
    # biases whole regions coherently, which is the realistic case.
    field = smooth_b1_field((128, 128), amplitude=0.2)
    step = np.abs(np.diff(field, axis=0)).max()
    assert step < 0.01


def test_zero_amplitude_gives_a_perfect_field():
    field = smooth_b1_field((16, 16), amplitude=0.0)
    np.testing.assert_allclose(field, 1.0, rtol=1e-12)


def test_b1_field_centre_is_jittered_when_given_a_generator():
    a = smooth_b1_field((32, 32), rng=np.random.default_rng(0))
    b = smooth_b1_field((32, 32), rng=np.random.default_rng(1))
    assert not np.allclose(a, b)
