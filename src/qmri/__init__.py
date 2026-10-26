"""Simulation and estimation tools for variable-flip-angle T1 mapping."""

from qmri.forward import (
    rician_noise,
    sigma_for_snr,
    smooth_b1_field,
    spgr_signal,
)
from qmri.dataset import (
    Dataset,
    build_splits,
    generate_split,
    normalise_per_voxel,
)
from qmri.fit import FitResult, despot1_linear, fit_image, fit_voxel
from qmri.phantom import Lesion, Phantom, insert_lesions, procedural_brain
from qmri.swin import SwinUNet

__all__ = [
    "spgr_signal",
    "rician_noise",
    "sigma_for_snr",
    "smooth_b1_field",
    "despot1_linear",
    "fit_voxel",
    "fit_image",
    "FitResult",
    "procedural_brain",
    "Phantom",
    "Lesion",
    "insert_lesions",
    "Dataset",
    "generate_split",
    "build_splits",
    "normalise_per_voxel",
    "SwinUNet",
]
