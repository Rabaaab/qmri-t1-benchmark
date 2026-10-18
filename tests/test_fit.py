"""Tests for the classical estimators.

The first test in this file is the one that decides whether the project is
safe to build on. Everything else -- phantoms, networks, lesion experiments
-- assumes that the forward model and the fit agree about the physics. If
they do not, every downstream number is meaningless, and it will still look
plausible.
"""

import numpy as np
import pytest

from qmri.fit import despot1_linear, fit_image, fit_voxel
from qmri.forward import rician_noise, sigma_for_snr, spgr_signal

ANGLES = [2.0, 5.0, 10.0, 15.0, 20.0]
TR = 15.0


# --- the test that matters -------------------------------------------------


@pytest.mark.parametrize("t1_true", [400.0, 832.0, 1331.0, 2500.0, 4000.0])
@pytest.mark.parametrize("b1_true", [0.70, 1.00, 1.25])
def test_nlls_recovers_t1_at_infinite_snr(t1_true, b1_true):
    """Noiseless round trip: simulate with T1, fit, get T1 back exactly.

    With no noise and a known B1 there is a single exact solution, so the
    only thing this can detect is a disagreement between the simulator and
    the fit -- a sign error, a degrees/radians slip, a TR in the wrong units.
    Those are exactly the bugs that produce beautiful, wrong results.
    """
    s0_true = 1234.0
    signals = spgr_signal(ANGLES, TR, t1_true, s0=s0_true, b1=b1_true)

    res = fit_voxel(signals, ANGLES, TR, b1=b1_true)

    assert res.success
    assert res.t1 == pytest.approx(t1_true, rel=1e-9)
    assert res.s0 == pytest.approx(s0_true, rel=1e-9)


@pytest.mark.parametrize("t1_true", [400.0, 832.0, 1331.0, 4000.0])
def test_despot1_is_also_exact_at_infinite_snr(t1_true):
    # The closed form is an independent derivation from the same equation,
    # so agreement between it and the nonlinear fit is a second, cheap check
    # on the algebra.
    signals = spgr_signal(ANGLES, TR, t1_true, s0=500.0, b1=1.0)
    t1, s0 = despot1_linear(signals, ANGLES, TR, b1=1.0)
    assert float(t1) == pytest.approx(t1_true, rel=1e-9)
    assert float(s0) == pytest.approx(500.0, rel=1e-9)


# --- B1: the failure the project is about ----------------------------------


@pytest.mark.parametrize("b1_true", [0.75, 0.85, 1.15, 1.25])
def test_ignoring_b1_biases_t1_substantially(b1_true):
    """Fitting as though the dial told the truth gives the wrong T1.

    This is the uncorrected baseline's whole weakness, and the error is
    large -- well beyond anything noise would explain -- which is why it
    shows up as a regional bias rather than as scatter.
    """
    t1_true = 832.0
    signals = spgr_signal(ANGLES, TR, t1_true, s0=1000.0, b1=b1_true)

    uncorrected = fit_voxel(signals, ANGLES, TR, b1=1.0)
    relative_error = abs(uncorrected.t1 - t1_true) / t1_true
    assert relative_error > 0.05


@pytest.mark.parametrize("b1_true", [0.75, 0.85, 1.15, 1.25])
def test_correcting_b1_removes_the_bias_entirely(b1_true):
    # Same data, same fit, correct B1 handed in: the bias is not a limit of
    # the estimator, it is missing information.
    t1_true = 832.0
    signals = spgr_signal(ANGLES, TR, t1_true, s0=1000.0, b1=b1_true)

    corrected = fit_voxel(signals, ANGLES, TR, b1=b1_true)
    assert corrected.t1 == pytest.approx(t1_true, rel=1e-9)


# --- behaviour under noise -------------------------------------------------


def test_fit_is_approximately_unbiased_at_high_snr():
    # Many noisy realisations of one voxel. Averaged, the estimate should
    # land close to the truth when the signal is well above the noise floor.
    rng = np.random.default_rng(0)
    t1_true, s0_true = 832.0, 1000.0
    clean = spgr_signal(ANGLES, TR, t1_true, s0=s0_true)
    sigma = s0_true / 300.0

    estimates = []
    for _ in range(150):
        res = fit_voxel(rician_noise(clean, sigma, rng), ANGLES, TR, b1=1.0)
        if res.success:
            estimates.append(res.t1)

    assert len(estimates) > 140
    assert np.mean(estimates) == pytest.approx(t1_true, rel=0.03)


def test_noise_increases_scatter():
    # Variance must grow with noise. If it does not, the noise is not
    # reaching the estimator and every SNR sweep in the study is a no-op.
    rng = np.random.default_rng(1)
    clean = spgr_signal(ANGLES, TR, 832.0, s0=1000.0)

    def scatter(sigma):
        vals = [
            fit_voxel(rician_noise(clean, sigma, rng), ANGLES, TR).t1
            for _ in range(120)
        ]
        return float(np.nanstd(vals))

    assert scatter(1000.0 / 500.0) < scatter(1000.0 / 50.0)


# --- the runtime / precision trade --------------------------------------


def test_default_tolerance_does_not_move_the_answer():
    """The default tolerance buys speed with precision nobody needs.

    A whole-image fit is tens of thousands of independent optimisations, so
    convergence tolerance is the single biggest lever on runtime. The
    default must still agree with an essentially exact solve far more
    closely than noise ever will.
    """
    signals = spgr_signal(ANGLES, TR, 832.0, s0=1000.0)

    exact = fit_voxel(signals, ANGLES, TR, tol=1e-14)
    default = fit_voxel(signals, ANGLES, TR)

    assert default.t1 == pytest.approx(exact.t1, rel=1e-8)


def test_default_tolerance_is_cheaper_than_a_tight_one():
    # The point of the default is fewer solver iterations. If this stops
    # being true the tolerance has stopped earning its place.
    rng = np.random.default_rng(5)
    clean = spgr_signal(ANGLES, TR, 1331.0, s0=1000.0)
    noisy = rician_noise(clean, sigma_for_snr(float(clean[2]), 100.0), rng)

    default = fit_voxel(noisy, ANGLES, TR)
    exact = fit_voxel(noisy, ANGLES, TR, tol=1e-14)

    assert default.nfev < exact.nfev
    # and the two answers still agree far inside the noise scatter
    assert default.t1 == pytest.approx(exact.t1, rel=1e-4)


# --- image-level plumbing --------------------------------------------------


def test_fit_image_recovers_a_noiseless_map():
    rng = np.random.default_rng(2)
    t1_map = rng.uniform(400.0, 3000.0, size=(4, 5))
    b1_map = rng.uniform(0.8, 1.2, size=(4, 5))
    stack = spgr_signal(ANGLES, TR, t1_map, s0=900.0, b1=b1_map)

    t1_hat, s0_hat = fit_image(stack, ANGLES, TR, b1=b1_map)

    np.testing.assert_allclose(t1_hat, t1_map, rtol=1e-7)
    np.testing.assert_allclose(s0_hat, 900.0, rtol=1e-7)


def test_fit_image_leaves_masked_out_voxels_as_nan():
    t1_map = np.full((3, 3), 832.0)
    stack = spgr_signal(ANGLES, TR, t1_map)
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 1] = True

    t1_hat, _ = fit_image(stack, ANGLES, TR, mask=mask)

    assert np.isfinite(t1_hat[1, 1])
    assert np.isnan(t1_hat[0, 0])
    assert np.count_nonzero(np.isfinite(t1_hat)) == 1


def test_fit_image_default_is_the_uncorrected_baseline():
    # Omitting b1 must mean "assume the dial told the truth", so that the
    # naive baseline is what you get by forgetting, not something you have
    # to opt into.
    t1_map = np.full((2, 2), 832.0)
    stack = spgr_signal(ANGLES, TR, t1_map, b1=0.8)

    default, _ = fit_image(stack, ANGLES, TR)
    explicit, _ = fit_image(stack, ANGLES, TR, b1=1.0)

    np.testing.assert_allclose(default, explicit, rtol=1e-12)
    assert abs(default[0, 0] - 832.0) / 832.0 > 0.05


# --- input validation ------------------------------------------------------


def test_fit_voxel_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="flip angles"):
        fit_voxel([1.0, 2.0, 3.0], [2.0, 5.0], TR)


def test_fit_voxel_needs_two_measurements():
    with pytest.raises(ValueError, match="at least two"):
        fit_voxel([1.0], [2.0], TR)


def test_despot1_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="flip angles were given"):
        despot1_linear(np.ones((3, 2)), [2.0, 5.0], TR)


def test_fit_image_rejects_a_bare_vector():
    with pytest.raises(ValueError, match="spatial axis"):
        fit_image(np.ones(5), ANGLES, TR)
