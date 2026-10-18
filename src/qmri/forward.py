"""The forward model: tissue properties in, images out.

Everything in this project rests on one equation. A spoiled gradient-echo
(SPGR / FLASH) sequence driven to steady state produces a magnitude signal

    S(alpha) = S0 * sin(a) * (1 - E1) / (1 - E1 * cos(a))

where

    E1 = exp(-TR / T1)
    a  = B1 * alpha

and the distinction between ``alpha`` and ``a`` is the entire reason B1
matters: ``alpha`` is the flip angle you requested from the scanner, ``a`` is
the one the spins actually experienced. The transmit field is not uniform
across a head, so B1 drifts smoothly away from 1.0 and a whole region of the
image is then systematically wrong.

Units: TR and T1 in milliseconds, flip angles in degrees. S0 is in arbitrary
units -- it absorbs proton density, receive-coil sensitivity and T2* decay at
the echo time, none of which this study needs to separate.

All functions broadcast, so the same call serves one voxel or a whole image
stack. Maps are indexed with the flip angle on axis 0: a return value of
shape ``(n_angles, H, W)`` for a ``(H, W)`` T1 map.
"""

from __future__ import annotations

import numpy as np

__all__ = ["spgr_signal", "rician_noise", "sigma_for_snr", "smooth_b1_field"]


def spgr_signal(flip_angles_deg, tr, t1, s0=1.0, b1=1.0):
    """Steady-state SPGR magnitude signal.

    Parameters
    ----------
    flip_angles_deg : 1-D array_like
        Nominal (requested) flip angles in degrees.
    tr : float
        Repetition time, milliseconds.
    t1 : float or ndarray
        True T1 in milliseconds. Voxels with ``t1 <= 0`` are treated as
        background and return exactly zero, which is what a phantom's
        air label should do.
    s0 : float or ndarray, optional
        Signal scaling. Broadcast against ``t1``.
    b1 : float or ndarray, optional
        Relative transmit field. ``1.0`` means the requested flip angle was
        delivered exactly; ``0.8`` means the tissue received 80% of it.

    Returns
    -------
    ndarray
        Shape ``(len(flip_angles_deg), *map_shape)``.
    """
    t1 = np.asarray(t1, dtype=float)
    s0 = np.asarray(s0, dtype=float)
    b1 = np.asarray(b1, dtype=float)

    alpha = np.deg2rad(np.asarray(flip_angles_deg, dtype=float))
    if alpha.ndim != 1:
        raise ValueError(
            f"flip_angles_deg must be 1-D, got shape {alpha.shape}"
        )
    if tr <= 0:
        raise ValueError(f"tr must be positive, got {tr}")

    # Put the flip angles on axis 0 and let the map dimensions broadcast.
    map_ndim = max(t1.ndim, s0.ndim, b1.ndim)
    alpha = alpha.reshape(-1, *([1] * map_ndim))

    a = b1 * alpha  # the flip angle the spins actually experienced

    valid = t1 > 0
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        e1 = np.exp(-tr / np.where(valid, t1, np.nan))
    e1 = np.where(valid, e1, 0.0)

    # 0 <= e1 < 1 and |cos a| <= 1, so the denominator cannot vanish.
    signal = s0 * np.sin(a) * (1.0 - e1) / (1.0 - e1 * np.cos(a))
    return np.where(valid, signal, 0.0)


def rician_noise(signal, sigma, rng):
    """Add realistic MRI magnitude noise.

    An MR image is reconstructed from a complex signal with roughly Gaussian
    noise in each channel, and then the magnitude is taken. That square root
    is the whole story: noise can push a magnitude *up* but never below zero,
    so in dark regions it does not merely scatter the values, it inflates
    them. Least-squares fitting assumes symmetric noise and therefore
    inherits that bias.

    Parameters
    ----------
    signal : array_like
        Noise-free magnitude signal.
    sigma : float
        Standard deviation of the Gaussian noise in each complex channel.
    rng : numpy.random.Generator
        Pass an explicit generator so every figure is reproducible.
    """
    signal = np.asarray(signal, dtype=float)
    if sigma < 0:
        raise ValueError(f"sigma must be non-negative, got {sigma}")
    if sigma == 0:
        return signal.copy()
    real = signal + rng.normal(0.0, sigma, size=signal.shape)
    imag = rng.normal(0.0, sigma, size=signal.shape)
    return np.hypot(real, imag)


def sigma_for_snr(reference_signal, snr):
    """Noise level giving ``snr`` at ``reference_signal``.

    SNR is meaningless without saying *where*. Pick one reference -- the
    white-matter signal at the flip angle that maximises it is the
    conventional choice -- and quote every noise level relative to it.
    """
    if snr <= 0:
        raise ValueError(f"snr must be positive, got {snr}")
    return float(reference_signal) / float(snr)


def smooth_b1_field(shape, amplitude=0.15, centre=None, rng=None):
    """A smooth, low-spatial-frequency transmit field centred near 1.0.

    Real B1 inhomogeneity is smooth and regional, never speckled -- that is
    precisely what makes it dangerous, because it biases whole anatomical
    regions coherently instead of averaging out. A quadratic bowl captures
    that character: the flip angle is close to correct near the centre of the
    field of view and progressively wrong toward the edges.

    Parameters
    ----------
    shape : tuple of int
        Spatial shape, e.g. ``(256, 256)``.
    amplitude : float
        Largest deviation from 1.0 anywhere in the field. ``0.15`` is a
        plausible 3 T brain; raise it for the out-of-distribution robustness
        test.

        The field is constructed to have mean *exactly* 1.0. That matters:
        a field whose mean drifted from 1.0 would impose a global scale
        error on top of the regional variation, and a global error is a
        confound rather than the effect under study. Because the geometry
        is asymmetric -- far more voxels lie near the centre than near a
        corner -- holding the mean at 1.0 makes the deviation asymmetric
        too, reaching ``-amplitude`` at the corners and roughly half that
        at the centre. Real transmit fields are asymmetric in just this way.
    centre : tuple of float, optional
        Bowl centre in fractional coordinates, ``(0.5, 0.5)`` being the image
        middle. Defaults to the middle.
    rng : numpy.random.Generator, optional
        If given and ``centre`` is not, the centre is jittered so that a
        training set contains a variety of fields rather than one repeated.
    """
    if amplitude < 0:
        raise ValueError(f"amplitude must be non-negative, got {amplitude}")

    if centre is None:
        if rng is None:
            centre = tuple(0.5 for _ in shape)
        else:
            centre = tuple(rng.uniform(0.35, 0.65) for _ in shape)

    axes = [
        (np.linspace(0.0, 1.0, n) - c) ** 2 for n, c in zip(shape, centre)
    ]
    r2 = np.zeros(shape, dtype=float)
    for axis, sq in enumerate(axes):
        bcast = [1] * len(shape)
        bcast[axis] = -1
        r2 = r2 + sq.reshape(bcast)

    # Centre the bowl on its own mean, then scale so the largest deviation
    # is exactly `amplitude`. This pins mean(B1) == 1.0 regardless of shape.
    deviation = r2.mean() - r2  # positive at the centre, negative at the edge
    peak = np.abs(deviation).max()
    if peak == 0.0:  # a 1-voxel image, or amplitude 0
        return np.ones(shape, dtype=float)
    return 1.0 + amplitude * deviation / peak
