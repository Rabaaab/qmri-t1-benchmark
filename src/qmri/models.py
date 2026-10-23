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

import warnings

import torch
from torch import nn

__all__ = [
    "T1_SCALE_MS",
    "to_scaled",
    "to_ms",
    "VoxelMLP",
    "UNet",
    "receptive_field_px",
]

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


class _DoubleConv(nn.Module):
    """Two 3x3 convolutions. The standard U-Net building block.

    Each 3x3 convolution lets a pixel see one step further in every
    direction. Stacking them is how the network accumulates a view of its
    neighbourhood -- and, in this study, how it acquires the ability to
    smooth a lesion away.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """Estimate a whole T1 map at once, using spatial context.

    The counterpart to :class:`VoxelMLP`. Same inputs, same targets, same
    training loop -- the single difference is that this one can see a
    pixel's neighbours. Any difference in the results is therefore
    attributable to spatial context and nothing else, which is the entire
    reason the MLP exists.

    **Depth is the independent variable, not a hyperparameter.** Each level
    halves the image, so a pixel at the bottom of a deep network summarises
    a large region of the original. That region *is* the distance over which
    the network can smooth, and smoothing is the mechanism that erases a
    small lesion. Varying ``depth`` and watching lesion retention is the
    ablation that turns "the CNN erased it" into "the CNN erased it
    *because* it was looking this far afield". Use
    :func:`receptive_field_px` to put a number on it.

    Parameters
    ----------
    in_channels : int
        One per flip angle, plus one for the B1 map.
    base_channels : int
        Width of the first level; doubles at each level down.
    depth : int
        Number of downsampling levels. 0 gives a purely local network with
        no downsampling at all, which is a useful extra control.
    """

    def __init__(
        self, in_channels: int = 6, base_channels: int = 32, depth: int = 3
    ):
        super().__init__()
        if depth < 0:
            raise ValueError(f"depth must be non-negative, got {depth}")
        if base_channels < 1:
            raise ValueError(
                f"base_channels must be at least 1, got {base_channels}"
            )

        self.depth = depth
        self.in_channels = in_channels

        widths = [base_channels * 2**i for i in range(depth + 1)]

        self.inc = _DoubleConv(in_channels, widths[0])
        self.downs = nn.ModuleList(
            _DoubleConv(widths[i], widths[i + 1]) for i in range(depth)
        )
        self.pool = nn.MaxPool2d(2)
        self.ups = nn.ModuleList(
            nn.ConvTranspose2d(widths[i + 1], widths[i], kernel_size=2, stride=2)
            for i in reversed(range(depth))
        )
        # After concatenating the skip connection the width doubles again.
        self.up_convs = nn.ModuleList(
            _DoubleConv(widths[i] * 2, widths[i]) for i in reversed(range(depth))
        )
        self.outc = nn.Conv2d(widths[0], 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(batch, channels, H, W)`` to ``(batch, H, W)`` scaled T1."""
        skips = []
        x = self.inc(x)
        for down in self.downs:
            skips.append(x)
            x = down(self.pool(x))

        for up, conv, skip in zip(self.ups, self.up_convs, reversed(skips)):
            x = up(x)
            # Odd image sizes leave the upsampled tensor a pixel short, so
            # pad before concatenating rather than demanding powers of two.
            dy, dx = skip.shape[-2] - x.shape[-2], skip.shape[-1] - x.shape[-1]
            if dy or dx:
                x = nn.functional.pad(
                    x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2]
                )
            x = conv(torch.cat([skip, x], dim=1))

        # Softplus for the same reason as the MLP: T1 cannot be negative,
        # and the classical fit is bounded the same way.
        return nn.functional.softplus(self.outc(x)).squeeze(1)

    @torch.no_grad()
    def predict_ms(self, x: torch.Tensor) -> torch.Tensor:
        """Predict a T1 map in milliseconds."""
        self.eval()
        return to_ms(self(x))


def receptive_field_px(
    model: nn.Module, size: int = 64, n_probes: int = 4, seed: int = 0
) -> int:
    """How far, in pixels, one output pixel can actually see.

    Measured rather than derived: push a gradient back from the centre
    output pixel and count how many input pixels it reaches. That is the
    exact region the network is able to draw on, which is also the distance
    over which it is able to smooth.

    This is the number that makes the depth ablation mean something. "Depth
    4 erased more lesions than depth 2" is an observation; "a network seeing
    61 pixels erased more than one seeing 13" is a mechanism.

    Probes are **random, not zeros**. With an all-zero input every ReLU sits
    exactly at its kink, where PyTorch defines the gradient as zero, so
    nothing propagates and the answer comes back as 0. Random probes keep
    the units active. Several are used and their reached pixels unioned,
    because any one probe only exercises the units that happen to be on for
    that input.

    Parameters
    ----------
    model : torch.nn.Module
        Any model taking ``(batch, channels, H, W)``.
    size : int
        Size of the probe image. Must comfortably exceed the receptive
        field; if it does not, the result is clipped by the image border
        and a warning is issued.
    n_probes : int
        Random probes to union over.
    seed : int
        For reproducibility.

    Returns
    -------
    int
        Width of the receptive field in pixels. Equal to ``size`` when the
        field is at least as large as the probe, in which case it is a lower
        bound and a warning is issued.
    """
    channels = getattr(model, "in_channels", 1)
    was_training = model.training
    model.eval()  # BatchNorm in train mode mixes the batch and inflates this

    generator = torch.Generator().manual_seed(seed)
    reached = torch.zeros(size, size, dtype=torch.bool)

    for _ in range(n_probes):
        probe = torch.randn(
            1, channels, size, size, generator=generator, requires_grad=True
        )
        out = model(probe)
        centre = out[0, size // 2, size // 2] if out.dim() == 3 else out[0, 0]
        model.zero_grad(set_to_none=True)
        centre.backward()
        reached |= probe.grad[0].abs().sum(dim=0) > 0

    if was_training:
        model.train()

    hits = reached.nonzero()
    if hits.numel() == 0:
        return 0
    rows, cols = hits[:, 0], hits[:, 1]
    width = int(max(rows.max() - rows.min(), cols.max() - cols.min()) + 1)

    if width >= size:
        warnings.warn(
            f"receptive field reached the {size}px probe border; {width} is a "
            "lower bound. Re-run with a larger `size` for the true value.",
            stacklevel=2,
        )
    return width
