"""Swin Transformer U-Net: the recent architecture in this comparison.

Written out rather than imported from a library, because the point is to be
able to explain it.

**Why a transformer here, scientifically.** This project measures what
*spatial context* does to small lesions. The three models form a ladder of
how far each one can see:

    voxel-wise fit / MLP   one pixel, no neighbours at all
    U-Net                  a local neighbourhood, set by depth
    Swin Transformer       every pixel in the window, and across windows

A convolution mixes a pixel with its immediate neighbours and builds reach
slowly by stacking layers. Attention lets every pixel in a window talk to
every other **in one step**. If spatial context is what erases a lesion,
the architecture with the most context should erase the most -- and that is
a prediction this ladder can test.

**How Swin differs from a plain Vision Transformer.** Attention over a
whole 128x128 image would mean 16,384 tokens attending to each other, which
costs memory in the square of that and is hopeless on a laptop. Swin's idea
is to run attention inside small **windows** (8x8 here), which is linear in
image size rather than quadratic. But windows alone would never let
information cross a window boundary, so alternate blocks **shift** the
window grid by half a window. Over two blocks, every pixel can reach
outside its original window. Window attention plus shifting is the whole
trick.

**Reading order**, which is also the order of this file:

1. ``window_partition`` / ``window_reverse`` -- cut the image into windows
   and put it back
2. ``WindowAttention`` -- attention inside one window
3. ``SwinBlock`` -- attention plus a small MLP, with the optional shift
4. ``PatchMerging`` / ``PatchExpand`` -- down and up, the U-shape
5. ``SwinUNet`` -- assembly

Reference: Liu et al., "Swin Transformer: Hierarchical Vision Transformer
using Shifted Windows", ICCV 2021. The U-shaped encoder-decoder arrangement
follows Cao et al., "Swin-Unet", ECCV workshops 2022.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = [
    "window_partition",
    "window_reverse",
    "WindowAttention",
    "SwinBlock",
    "PatchMerging",
    "PatchExpand",
    "SwinUNet",
]


def window_partition(x: torch.Tensor, window: int) -> torch.Tensor:
    """Cut ``(B, H, W, C)`` into non-overlapping windows.

    Returns ``(B * n_windows, window * window, C)`` -- every window becomes
    its own little sequence, and attention then runs on each independently.
    """
    b, h, w, c = x.shape
    x = x.view(b, h // window, window, w // window, window, c)
    # Bring the two window axes together so each window is contiguous.
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(-1, window * window, c)


def window_reverse(
    windows: torch.Tensor, window: int, h: int, w: int
) -> torch.Tensor:
    """Inverse of :func:`window_partition`."""
    b = windows.shape[0] // ((h // window) * (w // window))
    x = windows.view(b, h // window, w // window, window, window, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(b, h, w, -1)


class WindowAttention(nn.Module):
    """Multi-head self-attention inside one window.

    Each pixel produces three vectors: a **query** (what am I looking for),
    a **key** (what do I offer) and a **value** (what I pass on). Every
    pixel scores its query against every other pixel's key, softmaxes those
    scores into weights, and takes the weighted sum of their values. So a
    pixel's output is a blend of every pixel in its window, with the blend
    decided by content rather than fixed by a kernel.

    That blending is exactly the mechanism under investigation. A lesion
    that looks unlike its neighbours may be averaged away with them.

    **Relative position bias.** Attention by itself is blind to geometry --
    shuffle the pixels and the answer is unchanged. Swin adds a learned bias
    per (row offset, column offset) pair, so the model can learn that nearby
    pixels matter more. It is a lookup table of size
    ``(2*window-1) x (2*window-1)``, one entry per possible offset.
    """

    def __init__(self, dim: int, num_heads: int, window: int):
        super().__init__()
        if dim % num_heads:
            raise ValueError(
                f"dim {dim} must divide evenly into {num_heads} heads"
            )
        self.dim = dim
        self.num_heads = num_heads
        self.window = window
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

        self.relative_bias = nn.Parameter(
            torch.zeros((2 * window - 1) ** 2, num_heads)
        )
        nn.init.trunc_normal_(self.relative_bias, std=0.02)
        self.register_buffer(
            "relative_index", self._build_index(window), persistent=False
        )

    @staticmethod
    def _build_index(window: int) -> torch.Tensor:
        """Map every pixel pair in a window to its relative-offset slot."""
        coords = torch.stack(
            torch.meshgrid(
                torch.arange(window), torch.arange(window), indexing="ij"
            )
        ).flatten(1)
        relative = coords[:, :, None] - coords[:, None, :]
        relative = relative.permute(1, 2, 0).contiguous()
        relative += window - 1  # shift to non-negative
        relative[:, :, 0] *= 2 * window - 1
        return relative.sum(-1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """``x`` is ``(n_windows, tokens, dim)``."""
        nw, n, c = x.shape
        qkv = (
            self.qkv(x)
            .reshape(nw, n, 3, self.num_heads, c // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attention = (q * self.scale) @ k.transpose(-2, -1)

        bias = self.relative_bias[self.relative_index.view(-1)].view(
            n, n, self.num_heads
        )
        attention = attention + bias.permute(2, 0, 1).unsqueeze(0)

        if mask is not None:
            # Used only by shifted blocks: pixels that the cyclic roll made
            # neighbours without actually being adjacent are masked out.
            n_masks = mask.shape[0]
            attention = attention.view(
                nw // n_masks, n_masks, self.num_heads, n, n
            ) + mask.unsqueeze(1).unsqueeze(0)
            attention = attention.view(-1, self.num_heads, n, n)

        attention = attention.softmax(dim=-1)
        out = (attention @ v).transpose(1, 2).reshape(nw, n, c)
        return self.proj(out)


class SwinBlock(nn.Module):
    """One transformer block: attention, then a small MLP, both residual.

    ``shift=True`` rolls the image by half a window before partitioning, so
    this block's windows straddle the previous block's boundaries. Blocks
    are used in pairs, one unshifted and one shifted, which is how
    information escapes its window without ever paying the cost of global
    attention.

    LayerNorm comes *before* each sub-layer (pre-norm), which trains more
    stably than the original post-norm transformer.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window: int = 8,
        shift: bool = False,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.window = window
        self.shift = window // 2 if shift else 0

        self.norm1 = nn.LayerNorm(dim)
        self.attention = WindowAttention(dim, num_heads, window)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )

    def _attention_mask(self, h: int, w: int, device) -> torch.Tensor | None:
        """Stop rolled-around pixels attending to each other.

        The cyclic shift wraps the bottom edge to the top. Those pixels are
        now in the same window but are not actually neighbours, so they must
        not mix. Each region gets an id; pairs with different ids are given
        a large negative score, which softmax turns into zero weight.
        """
        if self.shift == 0:
            return None
        regions = torch.zeros((1, h, w, 1), device=device)
        slices = (
            slice(0, -self.window),
            slice(-self.window, -self.shift),
            slice(-self.shift, None),
        )
        count = 0
        for rows in slices:
            for cols in slices:
                regions[:, rows, cols, :] = count
                count += 1
        windows = window_partition(regions, self.window).squeeze(-1)
        mask = windows.unsqueeze(1) - windows.unsqueeze(2)
        return mask.masked_fill(mask != 0, -100.0).masked_fill(mask == 0, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` is ``(B, H, W, C)``."""
        b, h, w, c = x.shape
        shortcut = x
        x = self.norm1(x)

        # Pad so the window size divides the image exactly.
        pad_h = (self.window - h % self.window) % self.window
        pad_w = (self.window - w % self.window) % self.window
        if pad_h or pad_w:
            x = nn.functional.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        _, hp, wp, _ = x.shape

        if self.shift:
            x = torch.roll(x, (-self.shift, -self.shift), dims=(1, 2))

        windows = window_partition(x, self.window)
        attended = self.attention(
            windows, self._attention_mask(hp, wp, x.device)
        )
        x = window_reverse(attended, self.window, hp, wp)

        if self.shift:
            x = torch.roll(x, (self.shift, self.shift), dims=(1, 2))
        if pad_h or pad_w:
            x = x[:, :h, :w, :].contiguous()

        x = shortcut + x
        return x + self.mlp(self.norm2(x))


class PatchMerging(nn.Module):
    """Halve the resolution: glue each 2x2 block of pixels into one token.

    The transformer equivalent of a pooling layer. Four neighbouring tokens
    are concatenated (so channels go up 4x) and then projected back down to
    twice the original width. Coarser grid, richer tokens.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduce = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, h, w, c = x.shape
        if h % 2 or w % 2:
            x = nn.functional.pad(x, (0, 0, 0, w % 2, 0, h % 2))
            h, w = x.shape[1], x.shape[2]
        quad = torch.cat(
            [x[:, 0::2, 0::2], x[:, 1::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 1::2]],
            dim=-1,
        )
        return self.reduce(self.norm(quad))


class PatchExpand(nn.Module):
    """Double the resolution: split one token into a 2x2 block.

    The mirror of :class:`PatchMerging`, used on the way back up.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(dim // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, h, w, c = x.shape
        x = self.expand(x)  # (B, H, W, 2C)
        x = x.view(b, h, w, 2, 2, c // 2)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        return self.norm(x.view(b, h * 2, w * 2, c // 2))


class SwinUNet(nn.Module):
    """A U-shaped Swin Transformer that predicts a T1 map.

    Same inputs, same target and same training loop as :class:`qmri.models.UNet`
    -- so a difference in results is a difference in architecture, not in
    anything else.

    Shape of the thing: an input of ``(B, C, H, W)`` is cut into patches,
    passed through ``depth`` stages that each halve the resolution, then
    expanded back up with skip connections from the matching encoder stage,
    and finally projected to one channel per pixel.

    Parameters
    ----------
    in_channels : int
        One per flip angle, plus one for B1.
    embed_dim : int
        Token width at the finest stage; doubles at each stage down.
    depth : int
        Number of downsampling stages. As with the U-Net this controls how
        far the model can see, and is the independent variable.
    num_heads : int
        Attention heads at the finest stage; doubles alongside the width.
    window : int
        Window side length. Attention is computed inside these only.
    patch_size : int
        Pixels per token at the input. 2 keeps the map reasonably fine while
        quartering the token count.
    """

    def __init__(
        self,
        in_channels: int = 6,
        embed_dim: int = 32,
        depth: int = 2,
        num_heads: int = 4,
        window: int = 8,
        patch_size: int = 2,
    ):
        super().__init__()
        if depth < 0:
            raise ValueError(f"depth must be non-negative, got {depth}")
        if embed_dim % num_heads:
            raise ValueError(
                f"embed_dim {embed_dim} must divide into {num_heads} heads"
            )

        self.in_channels = in_channels
        self.depth = depth
        self.patch_size = patch_size

        # Patch embedding: a strided convolution is the standard way to turn
        # pixels into tokens.
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.patch_norm = nn.LayerNorm(embed_dim)

        dims = [embed_dim * 2**i for i in range(depth + 1)]
        heads = [num_heads * 2**i for i in range(depth + 1)]

        def stage(dim, head):
            # Blocks come in pairs: one plain, one shifted. That pairing is
            # what lets information cross window boundaries.
            return nn.ModuleList(
                [
                    SwinBlock(dim, head, window, shift=False),
                    SwinBlock(dim, head, window, shift=True),
                ]
            )

        self.encoder = nn.ModuleList(stage(dims[i], heads[i]) for i in range(depth))
        self.downs = nn.ModuleList(PatchMerging(dims[i]) for i in range(depth))
        self.bottleneck = stage(dims[depth], heads[depth])
        self.ups = nn.ModuleList(
            PatchExpand(dims[i + 1]) for i in reversed(range(depth))
        )
        # After concatenating the skip the width doubles, so project back.
        self.fuse = nn.ModuleList(
            nn.Linear(dims[i] * 2, dims[i]) for i in reversed(range(depth))
        )
        self.decoder = nn.ModuleList(
            stage(dims[i], heads[i]) for i in reversed(range(depth))
        )

        self.head = nn.Sequential(
            nn.LayerNorm(dims[0]), nn.Linear(dims[0], patch_size**2)
        )
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, C, H, W)`` in, ``(B, H, W)`` of scaled T1 out."""
        b, _, h, w = x.shape
        p = self.patch_size

        # Pad the image so it divides into whole patches and survives every
        # halving, then crop back at the very end.
        multiple = p * 2**self.depth
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple
        if pad_h or pad_w:
            x = nn.functional.pad(x, (0, pad_w, 0, pad_h))

        tokens = self.patch_embed(x).permute(0, 2, 3, 1)  # (B, H', W', C)
        tokens = self.patch_norm(tokens)

        skips = []
        for blocks, down in zip(self.encoder, self.downs):
            for block in blocks:
                tokens = block(tokens)
            skips.append(tokens)
            tokens = down(tokens)

        for block in self.bottleneck:
            tokens = block(tokens)

        for up, fuse, blocks, skip in zip(
            self.ups, self.fuse, self.decoder, reversed(skips)
        ):
            tokens = up(tokens)
            tokens = tokens[:, : skip.shape[1], : skip.shape[2], :]
            tokens = fuse(torch.cat([skip, tokens], dim=-1))
            for block in blocks:
                tokens = block(tokens)

        # Each token carries patch_size**2 values: unfold them back to pixels.
        out = self.head(tokens)
        bb, hh, ww, _ = out.shape
        out = out.view(bb, hh, ww, p, p).permute(0, 1, 3, 2, 4).contiguous()
        out = out.view(bb, hh * p, ww * p)
        out = out[:, :h, :w]

        # Softplus for the same reason as every other model here: T1 cannot
        # be negative, and the classical fit is bounded the same way.
        return nn.functional.softplus(out)

    @torch.no_grad()
    def predict_ms(self, x: torch.Tensor) -> torch.Tensor:
        """Predict a T1 map in milliseconds."""
        from qmri.models import to_ms

        self.eval()
        return to_ms(self(x))
