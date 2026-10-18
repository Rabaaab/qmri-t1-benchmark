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
- [ ] 2D digital phantom
- [ ] MLP
- [ ] U-Net
- [ ] Lesion retention experiment
- [ ] Out-of-distribution B1 test

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
