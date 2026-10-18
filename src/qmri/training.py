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

__all__ = ["TrainConfig", "History", "train"]


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


def train(
    model: nn.Module | Callable[[], nn.Module],
    train_x,
    train_y,
    val_x,
    val_y,
    config: TrainConfig = TrainConfig(),
    *,
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

    train_ds = TensorDataset(
        _as_tensor(train_x, device), _as_tensor(train_y, device)
    )
    loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)

    val_x_t = _as_tensor(val_x, device)
    val_y_t = _as_tensor(val_y, device)

    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = nn.MSELoss()

    history = History()
    best_state = copy.deepcopy(model.state_dict())
    since_improved = 0

    for epoch in range(config.epochs):
        model.train()
        running, seen = 0.0, 0
        for xb, yb in loader:
            optimiser.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimiser.step()
            running += float(loss.detach()) * len(xb)
            seen += len(xb)
        train_loss = running / max(seen, 1)

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(val_x_t), val_y_t))

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
