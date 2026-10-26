"""Loading every trained method, in one place.

Figures, the results database and any later analysis all need the same
thing: each method's T1 map over a test split. Doing that separately in
each script invites drift -- one of them quietly loads a model with
different settings, and two tables in the report stop agreeing for reasons
nobody can find.

So the loading lives here. Methods that have not been produced yet are
skipped rather than raising, so the project stays runnable while it grows:
with nothing trained you get the classical fits, and each model appears as
soon as its checkpoint exists.

The returned order is deliberate. It runs from no spatial context to the
most, which is the axis the whole study is organised around, and every
table and figure inherits that ordering for free.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from qmri.dataset import Dataset, scatter_to_maps, to_image_tensors, to_voxel_table
from qmri.models import UNet, VoxelMLP
from qmri.swin import SwinUNet

__all__ = ["METHOD_ORDER", "load_methods", "describe_methods"]

# No context -> local context -> global context.
METHOD_ORDER = (
    "Classical fit (no B1)",
    "Classical fit (+B1)",
    "MLP (no context)",
    "U-Net (local context)",
    "Swin (global context)",
)

# Voxel-wise methods see exactly one pixel. Recording that explicitly keeps
# them on the same axis as the networks rather than as a separate category.
VOXELWISE_FIELD = 1


def load_methods(
    test_set: Dataset,
    data_dir: str | Path,
    outputs_dir: str | Path,
) -> dict[str, np.ndarray]:
    """Every available method's T1 map over ``test_set``, in milliseconds.

    Parameters
    ----------
    test_set : Dataset
        The split to predict. Must match the cached classical fits.
    data_dir : path
        Holds ``baseline_<split>.npz`` from ``scripts/fit_baseline.py``.
    outputs_dir : path
        Holds the model checkpoints.

    Returns
    -------
    dict
        Method name to ``(N, H, W)`` map, ordered by :data:`METHOD_ORDER`.
    """
    data_dir, outputs_dir = Path(data_dir), Path(outputs_dir)
    found: dict[str, np.ndarray] = {}

    baseline = data_dir / "baseline_test.npz"
    if baseline.exists():
        with np.load(baseline) as f:
            found["Classical fit (no B1)"] = f["naive"]
            found["Classical fit (+B1)"] = f["corrected"]

    mlp_path = outputs_dir / "mlp.pt"
    if mlp_path.exists():
        table_x, _, table_mask = to_voxel_table(test_set)
        checkpoint = torch.load(mlp_path, weights_only=False)
        model = VoxelMLP(
            n_features=table_x.shape[1],
            hidden=checkpoint["config"].get("hidden", 256),
            depth=checkpoint["config"].get("depth", 3),
        )
        model.load_state_dict(checkpoint["state_dict"])
        with torch.no_grad():
            flat = model.predict_ms(torch.as_tensor(table_x)).numpy()
        found["MLP (no context)"] = scatter_to_maps(flat, table_mask)

    image_x, _, _ = to_image_tensors(test_set)
    for filename, label in [
        ("unet_d3.pt", "U-Net (local context)"),
        ("swin_d2.pt", "Swin (global context)"),
    ]:
        path = outputs_dir / filename
        if not path.exists():
            continue
        checkpoint = torch.load(path, weights_only=False)
        config = checkpoint["config"]
        if config["arch"] == "unet":
            model = UNet(
                in_channels=image_x.shape[1],
                base_channels=config["base_channels"],
                depth=config["depth"],
            )
        else:
            model = SwinUNet(
                in_channels=image_x.shape[1],
                embed_dim=config["embed_dim"],
                depth=config["depth"],
                num_heads=config["num_heads"],
                window=config["window"],
                patch_size=config["patch_size"],
            )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        # One image at a time: the whole split at once is needlessly large
        # in memory and buys nothing at inference.
        maps = np.empty_like(test_set.t1)
        with torch.no_grad():
            for i in range(len(test_set)):
                batch = torch.as_tensor(image_x[i : i + 1], dtype=torch.float32)
                maps[i] = model.predict_ms(batch).numpy()[0]
        found[label] = maps

    return {name: found[name] for name in METHOD_ORDER if name in found}


def describe_methods(outputs_dir: str | Path) -> dict[str, dict]:
    """Settings and training cost for each trained model.

    Separate from :func:`load_methods` because the database wants the
    provenance -- depth, parameter count, receptive field, how long it took
    -- while a figure only wants the numbers.
    """
    outputs_dir = Path(outputs_dir)
    out: dict[str, dict] = {
        "Classical fit (no B1)": {
            "receptive_field": VOXELWISE_FIELD,
            "settings": {"b1_correction": False},
        },
        "Classical fit (+B1)": {
            "receptive_field": VOXELWISE_FIELD,
            "settings": {"b1_correction": True},
        },
    }

    mlp_path = outputs_dir / "mlp.pt"
    if mlp_path.exists():
        checkpoint = torch.load(mlp_path, weights_only=False)
        config = checkpoint["config"]
        out["MLP (no context)"] = {
            "depth": config.get("depth"),
            "receptive_field": VOXELWISE_FIELD,
            "seed": config.get("seed"),
            "settings": config,
        }

    for filename, label in [
        ("unet_d3.pt", "U-Net (local context)"),
        ("swin_d2.pt", "Swin (global context)"),
    ]:
        path = outputs_dir / filename
        if not path.exists():
            continue
        checkpoint = torch.load(path, weights_only=False)
        config = checkpoint["config"]
        out[label] = {
            "depth": config.get("depth"),
            "n_params": checkpoint.get("n_params"),
            "receptive_field": checkpoint.get("receptive_field_px"),
            "seed": config.get("seed"),
            "best_val_loss": checkpoint.get("best_val_loss"),
            "settings": config,
        }

    return out
