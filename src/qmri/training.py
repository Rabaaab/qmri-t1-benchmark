"""A small, shared training loop.

Deliberately plain. The study compares what different architectures can
*see*, so the training procedure has to be the same for all of them --
otherwise a difference in results might just be a difference in how hard
each model was trained.

Early stopping is on validation loss, and the best weights are restored at
the end rather than the last ones kept. Without that, the reported model is
whichever epoch training happened to stop on.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

__all__ = ["TrainConfig", "History", "train", "masked_mse"]


@dataclass(frozen=True)
class TrainConfig:
    """Everything that controls a training run.

    Kept in one frozen object so a run can be recorded alongside its results
    and reproduced exactly.
    """

    epochs: int = 60
    batch_size: int = 4096
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 10
    seed: int = 0
    device: str = "cpu"


@dataclass
class History:
    """Loss curves and the outcome of a run."""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    best_epoch: int = -1
    best_val_loss: float = float("inf")
    stopped_early: bool = False

    @property
    def epochs_run(self) -> int:
        return len(self.train_loss)


def _as_tensor(array, device):
    return torch.as_tensor(np.asarray(array), dtype=torch.float32, device=device)


def masked_mse(prediction, target, mask=None):
    """Mean squared error over the masked elements only.

    Without a mask this is ordinary MSE. With one, background is excluded
    from both the numerator and the count, so a spatial model is scored on
    exactly the voxels a voxel-wise model was trained on.

    This is a fairness requirement, not a refinement. The voxel table drops
    background before the MLP ever sees it, so the MLP spends no capacity on
    air. Hand a U-Net whole images and more than half of every picture is
    background, where the correct answer is a trivial zero -- it would bank
    an easy win on those pixels, report a far lower loss, and the comparison
    between the two models would mean nothing.
    """
    squared = (prediction - target) ** 2
    if mask is None:
        return squared.mean()
    mask = mask.to(squared.dtype)
    return (squared * mask).sum() / mask.sum().clamp(min=1.0)


def train(
    model: nn.Module | Callable[[], nn.Module],
    train_x,
    train_y,
    val_x,
    val_y,
    config: TrainConfig = TrainConfig(),
    *,
    train_mask=None,
    val_mask=None,
    verbose: bool = True,
) -> tuple[nn.Module, History]:
    """Fit ``model``; return it with the best weights loaded, and its history.

    Inputs and targets are whatever the model expects -- a voxel table for
    the MLP, image stacks for a U-Net -- so the same loop serves both.
    Targets must already be in the model's output units; see
    :func:`qmri.models.to_scaled`.

    Parameters
    ----------
    model : torch.nn.Module, or a zero-argument callable returning one
        **Pass a factory**, e.g. ``lambda: VoxelMLP(hidden=256)``, to get a
        run that is fully determined by ``config.seed``.

        Weight initialisation consumes the global RNG, so a model built
        *before* this call is initialised from whatever state happened to be
        current -- which means two runs with the same seed can start from
        different weights. Given a factory, the model is constructed after
        seeding and the seed genuinely determines the whole run. A
        pre-built module is still accepted, for loading a checkpoint or
        resuming, but then only the batch shuffling is reproducible.
    train_x, train_y, val_x, val_y : array_like
        Training and validation data.
    config : TrainConfig
    train_mask, val_mask : array_like, optional
        Which elements count toward the loss. Required for a spatial model,
        so that background is excluded and both models are scored on the
        same voxels -- see :func:`masked_mse`.
    verbose : bool
        Print a line per epoch.

    Returns
    -------
    (model, history)
        The model is returned because a factory is constructed inside this
        function, so the caller would otherwise have no reference to it.
    """
    torch.manual_seed(config.seed)
    if not isinstance(model, nn.Module):
        model = model()  # constructed under the seed, so init is reproducible

    device = torch.device(config.device)
    model = model.to(device)

    tensors = [_as_tensor(train_x, device), _as_tensor(train_y, device)]
    if train_mask is not None:
        tensors.append(_as_tensor(train_mask, device))
    loader = DataLoader(
        TensorDataset(*tensors), batch_size=config.batch_size, shuffle=True
    )

    val_x_t = _as_tensor(val_x, device)
    val_y_t = _as_tensor(val_y, device)
    val_mask_t = None if val_mask is None else _as_tensor(val_mask, device)

    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history = History()
    best_state = copy.deepcopy(model.state_dict())
    since_improved = 0

    for epoch in range(config.epochs):
        model.train()
        running, seen = 0.0, 0
        for batch in loader:
            xb, yb = batch[0], batch[1]
            mb = batch[2] if len(batch) > 2 else None
            optimiser.zero_grad()
            loss = masked_mse(model(xb), yb, mb)
            loss.backward()
            optimiser.step()
            running += float(loss.detach()) * len(xb)
            seen += len(xb)
        train_loss = running / max(seen, 1)

        model.eval()
        with torch.no_grad():
            val_loss = float(masked_mse(model(val_x_t), val_y_t, val_mask_t))

        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)

        improved = val_loss < history.best_val_loss
        if improved:
            history.best_val_loss = val_loss
            history.best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            since_improved = 0
        else:
            since_improved += 1

        if verbose:
            mark = " *" if improved else ""
            print(
                f"  epoch {epoch + 1:>3}/{config.epochs}  "
                f"train {train_loss:.5f}  val {val_loss:.5f}{mark}"
            )

        if since_improved >= config.patience:
            history.stopped_early = True
            if verbose:
                print(
                    f"  no improvement for {config.patience} epochs, stopping"
                )
            break

    # Report the best model, not whichever epoch the loop happened to end on.
    model.load_state_dict(best_state)
    return model, history
