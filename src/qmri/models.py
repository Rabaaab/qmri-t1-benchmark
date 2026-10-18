"""Network architectures.

The MLP here is the control, and it is the part of this study most people
skip. It sees exactly what the classical fit sees -- one voxel's flip-angle
measurements, plus B1 -- and nothing else. No neighbours, no context.

That makes a three-way comparison interpretable. If the U-Net beats the
classical fit, the MLP tells you *why*: if the MLP also beats it, learning
was the ingredient; if only the U-Net does, the gain came from spatial
context, which is also the mechanism that can erase a lesion. Without the
MLP sitting in between, a U-Net win is uninterpretable.

Targets are trained in seconds rather than milliseconds (see
:data:`T1_SCALE_MS`). Squared error on numbers of order 1000 produces
gradients of order a million, which is a needless fight with the optimiser.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["T1_SCALE_MS", "to_scaled", "to_ms", "VoxelMLP"]

# Train on T1 in seconds. Keeps targets and gradients order 1.
T1_SCALE_MS = 1000.0


def to_scaled(t1_ms):
    """Milliseconds -> the units the networks are trained in."""
    return t1_ms / T1_SCALE_MS


def to_ms(scaled):
    """Network units -> milliseconds."""
    return scaled * T1_SCALE_MS


class VoxelMLP(nn.Module):
    """Estimate T1 from one voxel's measurements, with no spatial context.

    Deliberately ordinary: a few fully connected layers. The architecture is
    not the object of study, the *information available to it* is. Anything
    clever here would muddy the comparison it exists to support.

    The output passes through softplus, so predicted T1 is always positive.
    The classical fit is bounded the same way, so neither method is allowed
    to return a value that could not physically exist.

    Parameters
    ----------
    n_features : int
        Input width: one per flip angle, plus one for B1 if supplied.
    hidden : int
        Width of each hidden layer.
    depth : int
        Number of hidden layers.
    """

    def __init__(self, n_features: int = 6, hidden: int = 256, depth: int = 3):
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be at least 1, got {depth}")

        layers: list[nn.Module] = []
        width = n_features
        for _ in range(depth):
            layers += [nn.Linear(width, hidden), nn.ReLU()]
            width = hidden
        layers.append(nn.Linear(width, 1))

        self.net = nn.Sequential(*layers)
        self.n_features = n_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict T1 in scaled units, shape ``(batch,)``."""
        # Softplus keeps T1 positive without the hard zero gradient of ReLU.
        return nn.functional.softplus(self.net(x)).squeeze(-1)

    @torch.no_grad()
    def predict_ms(self, x: torch.Tensor) -> torch.Tensor:
        """Predict T1 in milliseconds."""
        self.eval()
        return to_ms(self(x))
