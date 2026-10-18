"""Voxel-wise T1 estimation: the classical baseline.

Two estimators live here.

``despot1_linear`` is the closed-form method. Divide the signal equation
through and it becomes a straight line, so T1 falls out of a linear
regression with no iteration at all. It is exact on noiseless data and
convenient, but noise appears in *both* regression variables, which biases
it. Its job in this project is to supply the starting guess for the real fit.

``fit_voxel`` is nonlinear least squares on the signal equation itself. This
is the baseline the networks have to beat, and the version that accepts a
measured B1 map is the honest one: if you pass ``b1=1.0`` while the true
field was 0.8, the fit will confidently return the wrong T1, which is
exactly the failure mode this project is about.

The critical property, enforced in ``tests/test_fit.py``: on noiseless data
with a known B1, ``fit_voxel`` must return the T1 that generated the signal
to machine precision. If it does not, the forward model and the fit disagree
about the physics and nothing built on top of them means anything.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.optimize import least_squares

from qmri.forward import spgr_signal

__all__ = ["despot1_linear", "fit_voxel", "fit_image", "FitResult"]

# Generous enough to hold any brain tissue at any field strength, so the
# optimiser never parks a solution on the boundary and calls it a fit.
T1_BOUNDS_MS = (1.0, 10_000.0)


class FitResult(NamedTuple):
    """Outcome of a single-voxel fit. T1 and S0 are NaN when the fit failed."""

    t1: float
    s0: float
    success: bool
    nfev: int


def despot1_linear(signals, flip_angles_deg, tr, b1=1.0):
    """Closed-form (DESPOT1) T1 estimate.

    Rearranging the SPGR equation gives

        S / sin(a) = E1 * (S / tan(a)) + S0 * (1 - E1)

    so regressing ``S/sin(a)`` on ``S/tan(a)`` yields slope ``E1``, and
    ``T1 = -TR / ln(E1)``.

    Parameters
    ----------
    signals : ndarray
        Shape ``(n_angles, *map_shape)``; the flip angle must be axis 0.
    flip_angles_deg : 1-D array_like
        Nominal flip angles in degrees, matching axis 0 of ``signals``.
    tr : float
        Repetition time, milliseconds.
    b1 : float or ndarray, optional
        Relative transmit field, broadcast against the map shape. Pass the
        true or measured field to correct for it; pass 1.0 to reproduce the
        uncorrected baseline.

    Returns
    -------
    (t1, s0) : tuple of ndarray
        Maps of shape ``map_shape``. Voxels whose regression slope falls
        outside ``(0, 1)`` carry no physical T1 and are returned as NaN.
    """
    signals = np.asarray(signals, dtype=float)
    alpha = np.deg2rad(np.asarray(flip_angles_deg, dtype=float))
    if alpha.ndim != 1:
        raise ValueError(f"flip_angles_deg must be 1-D, got {alpha.shape}")
    if signals.shape[0] != alpha.size:
        raise ValueError(
            f"signals has {signals.shape[0]} angles on axis 0 but "
            f"{alpha.size} flip angles were given"
        )
    if alpha.size < 2:
        raise ValueError("need at least two flip angles to fit a line")

    map_shape = signals.shape[1:]
    b1 = np.asarray(b1, dtype=float)
    a = b1 * alpha.reshape(-1, *([1] * len(map_shape)))

    with np.errstate(divide="ignore", invalid="ignore"):
        x = signals / np.tan(a)
        y = signals / np.sin(a)

    n = alpha.size
    sx = x.sum(axis=0)
    sy = y.sum(axis=0)
    sxx = (x * x).sum(axis=0)
    sxy = (x * y).sum(axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
        intercept = (sy - slope * sx) / n
        t1 = -tr / np.log(slope)
        s0 = intercept / (1.0 - slope)

    # E1 must lie strictly inside (0, 1) for T1 to be positive and finite.
    physical = np.isfinite(slope) & (slope > 0.0) & (slope < 1.0)
    t1 = np.where(physical, t1, np.nan)
    s0 = np.where(physical, s0, np.nan)
    return t1, s0


def fit_voxel(
    signals,
    flip_angles_deg,
    tr,
    b1=1.0,
    t1_init=None,
    s0_init=None,
    t1_bounds=T1_BOUNDS_MS,
):
    """Nonlinear least-squares T1 and S0 for a single voxel.

    Parameters
    ----------
    signals : 1-D array_like
        One measurement per flip angle.
    flip_angles_deg : 1-D array_like
        Nominal flip angles in degrees.
    tr : float
        Repetition time, milliseconds.
    b1 : float, optional
        Relative transmit field assumed by the fit. This is the knob that
        separates the corrected baseline from the uncorrected one.
    t1_init, s0_init : float, optional
        Starting guesses. Taken from :func:`despot1_linear` when omitted,
        which is both faster and more robust than a fixed starting point.
    t1_bounds : (float, float), optional
        Search interval for T1 in milliseconds.

    Returns
    -------
    FitResult
    """
    signals = np.asarray(signals, dtype=float).ravel()
    alpha = np.asarray(flip_angles_deg, dtype=float).ravel()
    if signals.size != alpha.size:
        raise ValueError(
            f"got {signals.size} signals for {alpha.size} flip angles"
        )
    if signals.size < 2:
        raise ValueError("need at least two measurements to fit two parameters")

    if t1_init is None or s0_init is None:
        guess_t1, guess_s0 = despot1_linear(signals, alpha, tr, b1=b1)
        if t1_init is None:
            t1_init = float(guess_t1)
        if s0_init is None:
            s0_init = float(guess_s0)

    # The closed form returns NaN on noise-dominated voxels. Fall back to a
    # mid-range brain T1 and the largest observed signal rather than give up.
    lo, hi = float(t1_bounds[0]), float(t1_bounds[1])
    if not np.isfinite(t1_init) or not (lo < t1_init < hi):
        t1_init = 1000.0
    if not np.isfinite(s0_init) or s0_init <= 0.0:
        s0_init = max(float(np.max(np.abs(signals))), 1e-12)

    def residual(params):
        t1, s0 = params
        return spgr_signal(alpha, tr, t1, s0=s0, b1=b1) - signals

    try:
        out = least_squares(
            residual,
            x0=[t1_init, s0_init],
            bounds=([lo, 0.0], [hi, np.inf]),
            method="trf",
            xtol=1e-14,
            ftol=1e-14,
            gtol=1e-14,
        )
    except (ValueError, FloatingPointError):
        return FitResult(np.nan, np.nan, False, 0)

    if not out.success:
        return FitResult(np.nan, np.nan, False, int(out.nfev))
    return FitResult(float(out.x[0]), float(out.x[1]), True, int(out.nfev))


def fit_image(signals, flip_angles_deg, tr, b1=None, mask=None, **kwargs):
    """Fit every voxel of an image stack independently.

    This is the limitation the whole project interrogates, made explicit: the
    loop body never looks at a neighbour. A 256x256 slice is 65,536
    independent two-parameter optimisations, and it is slow -- expect tens of
    seconds. That cost is itself a result worth reporting, since a trained
    network evaluates in milliseconds.

    Parameters
    ----------
    signals : ndarray
        Shape ``(n_angles, *map_shape)``.
    flip_angles_deg : 1-D array_like
        Nominal flip angles in degrees.
    tr : float
        Repetition time, milliseconds.
    b1 : ndarray or float, optional
        Transmit field assumed by the fit, broadcast to ``map_shape``.
        Defaults to 1.0 everywhere, i.e. the uncorrected baseline.
    mask : ndarray of bool, optional
        Voxels to fit. Everything else is left NaN. Skipping the background
        is worth a large constant factor.
    **kwargs
        Passed to :func:`fit_voxel`.

    Returns
    -------
    (t1, s0) : tuple of ndarray
        Maps of shape ``map_shape``, NaN outside the mask or where the fit
        failed.
    """
    signals = np.asarray(signals, dtype=float)
    if signals.ndim < 2:
        raise ValueError(
            "signals must have the flip angle on axis 0 and at least one "
            f"spatial axis, got shape {signals.shape}"
        )
    map_shape = signals.shape[1:]
    n_vox = int(np.prod(map_shape))
    flat = signals.reshape(signals.shape[0], n_vox)

    b1_flat = np.broadcast_to(
        np.asarray(1.0 if b1 is None else b1, dtype=float), map_shape
    ).reshape(n_vox)

    if mask is None:
        mask_flat = np.ones(n_vox, dtype=bool)
    else:
        mask_flat = np.broadcast_to(
            np.asarray(mask, dtype=bool), map_shape
        ).reshape(n_vox)

    t1 = np.full(n_vox, np.nan)
    s0 = np.full(n_vox, np.nan)
    for i in np.flatnonzero(mask_flat):
        res = fit_voxel(
            flat[:, i], flip_angles_deg, tr, b1=float(b1_flat[i]), **kwargs
        )
        if res.success:
            t1[i] = res.t1
            s0[i] = res.s0

    return t1.reshape(map_shape), s0.reshape(map_shape)
