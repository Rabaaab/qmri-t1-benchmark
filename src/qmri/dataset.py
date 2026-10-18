"""Building the training, validation and test sets.

Two decisions in this module matter more than any hyperparameter chosen
later, because getting either wrong produces results that look excellent and
mean nothing.

**One phantom is one subject.** Splits are by phantom, never by slice or by
voxel. Adjacent slices of the same head are nearly identical, so a
slice-level split leaks the test set into training and every reported error
comes out flatteringly low. Each phantom here gets its own anatomy from its
own spawned seed stream, and a phantom belongs to exactly one split.

**Lesions appear in the test set only.** This is the entire experiment. A
network shown lesions during training learns to preserve them, which answers
a much easier question than the one being asked: what happens to a small
structure the model was never given a reason to keep? Training on clean
anatomy and testing on lesioned anatomy is what makes erasure observable.
:func:`build_splits` enforces this rather than leaving it to a caller to
remember.

Scale is removed from the inputs on purpose -- see :func:`normalise_per_voxel`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from qmri.forward import rician_noise, sigma_for_snr, smooth_b1_field, spgr_signal
from qmri.phantom import Lesion, insert_lesions, procedural_brain

__all__ = [
    "Dataset",
    "generate_split",
    "build_splits",
    "normalise_per_voxel",
    "to_voxel_table",
    "scatter_to_maps",
]

DEFAULT_ANGLES = (2.0, 5.0, 10.0, 15.0, 20.0)
DEFAULT_TR = 15.0
DEFAULT_SNR = 100.0
DEFAULT_B1_AMPLITUDE = 0.15

# Flat record of every inserted lesion, so the whole test set's lesions
# survive a round trip through a single .npz without pickling.
_LESION_DTYPE = np.dtype(
    [
        ("phantom", np.int32),
        ("row", np.int32),
        ("col", np.int32),
        ("radius_px", np.float32),
        ("t1_ms", np.float32),
    ]
)


@dataclass(frozen=True)
class Dataset:
    """Simulated acquisitions and the ground truth that produced them.

    Attributes
    ----------
    images : ndarray, shape (N, n_angles, H, W)
        Noisy magnitude images. This is the network input.
    b1 : ndarray, shape (N, H, W)
        True transmit field. Handed to the networks as an extra channel,
        because the corrected classical baseline gets it too and the
        comparison has to be fair.
    t1 : ndarray, shape (N, H, W)
        True T1 in milliseconds. The target.
    labels : ndarray, shape (N, H, W)
        Tissue labels, for per-tissue error statistics. Never an input.
    lesions : ndarray, structured
        One row per inserted lesion, with the phantom it belongs to. Empty
        for training and validation splits.
    """

    images: np.ndarray
    b1: np.ndarray
    t1: np.ndarray
    labels: np.ndarray
    lesions: np.ndarray
    angles: np.ndarray
    tr: float
    snr: float

    def __len__(self) -> int:
        return int(self.images.shape[0])

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.images.shape[2]), int(self.images.shape[3]))

    @property
    def mask(self) -> np.ndarray:
        """Tissue voxels, i.e. everything worth scoring."""
        return self.labels > 0

    def lesions_for(self, index: int) -> list[Lesion]:
        """The lesions belonging to one phantom, as records."""
        rows = self.lesions[self.lesions["phantom"] == index]
        return [
            Lesion(
                row=int(r["row"]),
                col=int(r["col"]),
                radius_px=float(r["radius_px"]),
                t1_ms=float(r["t1_ms"]),
            )
            for r in rows
        ]

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            images=self.images,
            b1=self.b1,
            t1=self.t1,
            labels=self.labels,
            lesions=self.lesions,
            angles=self.angles,
            tr=np.float64(self.tr),
            snr=np.float64(self.snr),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> Dataset:
        with np.load(Path(path)) as f:
            return cls(
                images=f["images"],
                b1=f["b1"],
                t1=f["t1"],
                labels=f["labels"],
                lesions=f["lesions"],
                angles=f["angles"],
                tr=float(f["tr"]),
                snr=float(f["snr"]),
            )

    def summary(self) -> str:
        h, w = self.shape
        mb = sum(
            a.nbytes for a in (self.images, self.b1, self.t1, self.labels)
        ) / 1024**2
        return (
            f"{len(self)} phantoms, {h}x{w}, {len(self.angles)} angles, "
            f"SNR {self.snr:.0f}, {len(self.lesions)} lesions, {mb:.1f} MB"
        )


def normalise_per_voxel(images: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Divide each voxel's flip-angle vector by its own norm.

    T1 is encoded in the *shape* of the signal-versus-flip-angle curve, not
    in its height: S0 absorbs proton density, coil sensitivity and T2*
    decay, none of which carry T1 information. A network fed raw magnitudes
    can therefore shortcut -- learn which tissue a brightness corresponds
    to, then recall that tissue's T1 -- which is memorising anatomy, not
    estimating a physical parameter, and it collapses on a lesion whose
    brightness it has never seen.

    Removing the scale forces the network to use the same information the
    classical fit uses, which is what makes the comparison meaningful.

    Parameters
    ----------
    images : ndarray, shape (N, n_angles, H, W)
    eps : float
        Floor on the norm, so background voxels give zero rather than NaN.
    """
    images = np.asarray(images, dtype=np.float32)
    norm = np.linalg.norm(images, axis=1, keepdims=True)
    return np.divide(images, np.maximum(norm, eps), dtype=np.float32)


def generate_split(
    n: int,
    shape: tuple[int, int] = (128, 128),
    seed_sequence: np.random.SeedSequence | None = None,
    *,
    with_lesions: bool = False,
    n_lesions: int = 4,
    snr: float = DEFAULT_SNR,
    b1_amplitude: float = DEFAULT_B1_AMPLITUDE,
    angles=DEFAULT_ANGLES,
    tr: float = DEFAULT_TR,
) -> Dataset:
    """Simulate ``n`` independent phantoms and their acquisitions.

    Each phantom draws from its own spawned generator, so phantom ``i`` is
    identical whether you generate 10 phantoms or 1000 -- which means a
    split can be extended later without invalidating what came before.

    Parameters
    ----------
    n : int
        Number of phantoms.
    shape : (int, int)
        Image shape.
    seed_sequence : numpy.random.SeedSequence, optional
        Root of this split's randomness. Use :func:`build_splits` to get
        provably independent ones.
    with_lesions : bool
        Insert lesions. Should be true only for a test split; see the module
        docstring.
    n_lesions : int
        Lesions attempted per phantom, when enabled.
    snr : float
        Signal-to-noise ratio at the white-matter signal.
    b1_amplitude : float
        Transmit field inhomogeneity. Raise it for the out-of-distribution
        robustness test.
    angles : sequence of float
        Nominal flip angles in degrees.
    tr : float
        Repetition time in milliseconds.

    Returns
    -------
    Dataset
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    if seed_sequence is None:
        seed_sequence = np.random.SeedSequence()

    angles = np.asarray(angles, dtype=float)
    n_angles = angles.size

    images = np.empty((n, n_angles, *shape), dtype=np.float32)
    b1_all = np.empty((n, *shape), dtype=np.float32)
    t1_all = np.empty((n, *shape), dtype=np.float32)
    labels_all = np.empty((n, *shape), dtype=np.int8)
    lesion_rows: list[tuple] = []

    for i, child in enumerate(seed_sequence.spawn(n)):
        rng = np.random.default_rng(child)

        phantom = procedural_brain(shape, rng=rng, jitter_anatomy=True)
        if with_lesions:
            phantom, placed = insert_lesions(phantom, rng, n_lesions=n_lesions)
            lesion_rows.extend(
                (i, le.row, le.col, le.radius_px, le.t1_ms) for le in placed
            )

        b1 = smooth_b1_field(shape, amplitude=b1_amplitude, rng=rng)
        clean = spgr_signal(angles, tr, phantom.t1, s0=phantom.pd, b1=b1)

        # Quote SNR against one fixed reference per phantom: the median
        # white-matter signal near its Ernst angle. Anything else makes the
        # noise level depend on how much CSF happens to be in the slice.
        white = phantom.labels == 1
        reference = float(np.median(clean[2][white]))
        images[i] = rician_noise(clean, sigma_for_snr(reference, snr), rng)

        b1_all[i] = b1
        t1_all[i] = phantom.t1
        labels_all[i] = phantom.labels

    return Dataset(
        images=images,
        b1=b1_all,
        t1=t1_all,
        labels=labels_all,
        lesions=np.array(lesion_rows, dtype=_LESION_DTYPE),
        angles=angles,
        tr=float(tr),
        snr=float(snr),
    )


def build_splits(
    n_train: int = 120,
    n_val: int = 20,
    n_test: int = 40,
    shape: tuple[int, int] = (128, 128),
    seed: int = 0,
    *,
    n_lesions: int = 4,
    **kwargs,
) -> dict[str, Dataset]:
    """Build train, validation and test splits that cannot leak into each other.

    The three splits draw from independent child streams of one root seed,
    so no phantom can appear in two splits, and the whole set is reproducible
    from ``seed`` alone.

    Lesions are inserted into the test split and nowhere else. That is not a
    default to be overridden, it is the experiment: a model that trained on
    lesions would have learned to keep them.

    Returns
    -------
    dict
        ``{"train": Dataset, "val": Dataset, "test": Dataset}``.
    """
    train_ss, val_ss, test_ss = np.random.SeedSequence(seed).spawn(3)
    return {
        "train": generate_split(
            n_train, shape, train_ss, with_lesions=False, **kwargs
        ),
        "val": generate_split(n_val, shape, val_ss, with_lesions=False, **kwargs),
        "test": generate_split(
            n_test,
            shape,
            test_ss,
            with_lesions=True,
            n_lesions=n_lesions,
            **kwargs,
        ),
    }


def to_voxel_table(
    dataset: Dataset, *, normalise: bool = True, include_b1: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten a dataset into a per-voxel feature table.

    This is the MLP's view of the world, and the point of the MLP: it sees
    one voxel at a time, exactly like the classical fit, so any advantage it
    shows over that fit comes from *learning* rather than from spatial
    context. The U-Net gets the images instead, and the difference between
    the two is the contribution of context.

    Background voxels are dropped. They carry no T1 and would otherwise be
    most of the training set.

    Parameters
    ----------
    dataset : Dataset
    normalise : bool
        Scale-normalise each voxel's flip-angle vector. See
        :func:`normalise_per_voxel` for why this matters.
    include_b1 : bool
        Append the transmit field as a feature. On by default, because the
        corrected classical baseline is handed B1 and the comparison has to
        be fair.

    Returns
    -------
    (features, targets, mask)
        ``features`` has shape ``(n_voxels, n_angles [+ 1])``, ``targets``
        is T1 in milliseconds, and ``mask`` is the ``(N, H, W)`` boolean
        array needed to put predictions back on a grid. The row order of
        ``features`` matches ``np.flatnonzero(mask)``.
    """
    images = (
        normalise_per_voxel(dataset.images)
        if normalise
        else dataset.images.astype(np.float32)
    )
    mask = dataset.mask

    # (N, A, H, W) -> (n_voxels, A), keeping only tissue.
    features = images.transpose(0, 2, 3, 1)[mask]
    if include_b1:
        features = np.concatenate(
            [features, dataset.b1[mask][:, None].astype(np.float32)], axis=1
        )

    targets = dataset.t1[mask].astype(np.float32)
    return np.ascontiguousarray(features), targets, mask


def scatter_to_maps(
    values: np.ndarray, mask: np.ndarray, fill: float = np.nan
) -> np.ndarray:
    """Put per-voxel predictions back onto the image grid.

    Inverse of the flattening done by :func:`to_voxel_table`; ``values`` must
    be in the same order its ``features`` were.
    """
    values = np.asarray(values).ravel()
    n_expected = int(np.count_nonzero(mask))
    if values.size != n_expected:
        raise ValueError(
            f"got {values.size} values for {n_expected} masked voxels"
        )
    maps = np.full(mask.shape, fill, dtype=np.float32)
    maps[mask] = values
    return maps
