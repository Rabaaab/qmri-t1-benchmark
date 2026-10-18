"""Nominal tissue properties for the digital phantom, at 3 T.

A warning worth carrying into the report: the literature does not agree on
these numbers. Wansapura (1999) and Stanisz (2005) differ by roughly 25% for
white matter and 35% for grey matter, both measured at 3 T on healthy
volunteers. That disagreement is not sloppiness, it is the very problem
quantitative MRI exists to solve, and it is why this project measures a
*bias* against a known simulated truth rather than against a textbook value.

For a simulation study the exact choice barely matters -- what matters is
that the truth is known exactly, and that whichever values you pick are
stated and cited. Do not quote these numbers as measurements.

Sources
-------
Wansapura JP, Holland SK, Dunn RS, Ball WS Jr.
    "NMR relaxation times in the human brain at 3.0 Tesla."
    J Magn Reson Imaging. 1999;9(4):531-538.
Stanisz GJ, Odrobina EE, Pun J, et al.
    "T1, T2 relaxation and magnetization transfer in tissue at 3T."
    Magn Reson Med. 2005;54(3):507-512.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Tissue", "TISSUES_3T", "LESION_3T", "BACKGROUND"]


@dataclass(frozen=True)
class Tissue:
    """One tissue class in the phantom.

    Attributes
    ----------
    name : str
        Human-readable label, used in figures and per-tissue result tables.
    label : int
        Integer value identifying this tissue in the phantom label map.
    t1_ms : float
        Nominal T1 in milliseconds.
    t1_spread_ms : float
        Standard deviation of within-tissue T1 variation. Real tissue is not
        a single number, and a phantom of perfectly uniform regions would
        flatter any method that smooths.
    pd : float
        Relative proton density, scaling S0. Arbitrary units.
    """

    name: str
    label: int
    t1_ms: float
    t1_spread_ms: float
    pd: float


# Background / air. Zero T1 is the sentinel the forward model recognises,
# returning exactly zero signal.
BACKGROUND = Tissue("background", label=0, t1_ms=0.0, t1_spread_ms=0.0, pd=0.0)

# Wansapura et al. 1999, Table 1, 3.0 T. CSF is not tabulated there and is
# taken as a round 4000 ms, which is within the usual 3.7-4.5 s range.
TISSUES_3T: tuple[Tissue, ...] = (
    BACKGROUND,
    Tissue("white matter", label=1, t1_ms=832.0, t1_spread_ms=30.0, pd=0.70),
    Tissue("grey matter", label=2, t1_ms=1331.0, t1_spread_ms=60.0, pd=0.80),
    Tissue("csf", label=3, t1_ms=4000.0, t1_spread_ms=150.0, pd=1.00),
)

# A demyelinating lesion sitting in white matter: T1 elevated toward grey
# matter values, proton density slightly raised. The contrast is deliberately
# modest -- an obvious lesion would be a soft test, and the question is
# whether a method erases the *subtle* ones.
LESION_3T = Tissue("lesion", label=4, t1_ms=1200.0, t1_spread_ms=40.0, pd=0.75)
