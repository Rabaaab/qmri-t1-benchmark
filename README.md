# qmri-t1-benchmark

Do convolutional networks estimate T1 more accurately than voxel-wise
fitting — or do they just smooth away the lesions we are looking for?

## The question

Quantitative MRI aims to measure a physical tissue property (here T1, the
spin-lattice relaxation time) rather than produce a picture whose brightness
is arbitrary. The standard estimator fits a physical signal model to a handful
of images **one voxel at a time, independently**. A CNN can use spatial
context instead, which should help — but a network trained on normal anatomy
has every incentive to smooth a 3-voxel lesion out of existence, and it will
still report a lower overall error when it does.

This repository measures that trade-off on simulated data, where the true T1
is known exactly.

## The design

Four estimators, chosen so that a win can be attributed:

| Estimator | Sees | Isolates |
|---|---|---|
| NLLS, no B1 correction | one voxel | naive baseline |
| NLLS, with B1 correction | one voxel | the baseline to beat |
| MLP | one voxel | learning, without spatial context |
| U-Net | whole image | learning **plus** spatial context |

The MLP is the control. Without it, a U-Net win cannot be separated into
"learning helped" versus "smoothing helped".

Three measurements: bias and variance per tissue; lesion retention as a
function of lesion size; and robustness when tested on B1 errors more severe
than those seen in training.

## Status

- [x] Forward model (SPGR / variable flip angle, with B1 and Rician noise)
- [x] Voxel-wise NLLS fit, with and without B1 correction
- [x] Verification: noiseless round-trip recovers T1 to machine precision
- [x] 2D digital phantom, and the first end-to-end result (below)
- [ ] MLP
- [ ] U-Net
- [ ] Lesion retention experiment
- [ ] Out-of-distribution B1 test

## First result

128x128 phantom, 5 flip angles, TR 15 ms, SNR 100, B1 spanning 0.85 to 1.08.
The same five images fitted twice, scored against the known truth (ms):

| tissue | true T1 | bias, B1 ignored | SD | bias, B1 corrected | SD |
|---|---|---|---|---|---|
| white matter | 831 | **+96.0** | 30.4 | +0.1 | 20.2 |
| grey matter | 1332 | **+68.4** | 48.3 | +0.2 | 33.8 |
| CSF | 4003 | **+162.2** | 289.7 | +4.0 | 142.5 |

Ignoring B1 inflates white-matter T1 by 11.5%, which is larger than the
difference between the Wansapura and Stanisz literature values for the same
tissue. Correcting it removes the bias essentially completely, leaving only
noise-driven scatter -- so the error is missing information, not a limit of
the estimator.

The residual SD is the interesting part, and the reason the networks might
win: it is what spatial context could in principle suppress. CSF is the
hardest case by far, because a long T1 is poorly constrained by these flip
angles.

## Install

```
python -m pip install -r requirements.txt
```

## Verify the simulator before trusting anything built on it

```
python -m pytest -q
```

The test that matters is `test_nlls_recovers_t1_at_infinite_snr`. With no
noise and a known B1, the fit must return the T1 that generated the signal to
~1e-9 relative error. If it does not, the forward model and the fit disagree
about the physics and every downstream result is meaningless.

## Reproduce the Day 1 figure

```
python scripts/day1_signal_curve.py
```

Writes `outputs/day1_signal_curve.png`: signal against flip angle for several
T1 values, which is the curve the whole method rests on.

## Reproduce the Day 3 figure

```
python scripts/day3_phantom_fit.py
```

Writes `outputs/day3_phantom_fit.png` and prints the table above. Takes about
two minutes: a 128x128 slice is 8,848 voxels and each is fitted twice, as
17,696 completely independent optimisations that never look at a neighbour.
That cost is itself a result worth reporting, since a trained network
evaluates in milliseconds.

If the fitted map resembles the ground truth, the project is safe to build
on and everything after this is ordinary engineering.

## Layout

```
src/qmri/forward.py   signal model, B1, Rician noise
src/qmri/fit.py       DESPOT1 closed form + nonlinear least squares
tests/                correctness of the above
scripts/              one runnable script per figure
data/                 phantoms (gitignored)
outputs/              figures and results (gitignored)
```

## Notes on values

T1 values used in the phantom are approximate literature figures for 3 T and
are cited at the point of use in `src/qmri/tissues.py`. Primary sources to
cite in the report: Wansapura et al., *JMRI* 1999; Stanisz et al., *MRM* 2005.
