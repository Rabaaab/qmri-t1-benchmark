"""Tests for the Swin Transformer.

Two groups. The first checks the mechanics that are easy to get subtly
wrong -- window partitioning, the shift, the attention mask -- where a bug
would not crash anything, it would just quietly train a worse model. The
second checks the property the experiment depends on: that attention really
does give a wider view than convolution at the same depth.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from qmri.models import UNet, receptive_field_px
from qmri.swin import (
    PatchExpand,
    PatchMerging,
    SwinBlock,
    SwinUNet,
    WindowAttention,
    window_partition,
    window_reverse,
)


# --- windowing -------------------------------------------------------------


def test_partition_and_reverse_round_trip():
    # If these two disagree, pixels silently end up in the wrong place and
    # the model still trains -- just on scrambled data.
    x = torch.randn(2, 16, 16, 8)
    back = window_reverse(window_partition(x, 4), 4, 16, 16)
    torch.testing.assert_close(back, x)


def test_partition_makes_the_expected_number_of_windows():
    x = torch.randn(3, 16, 24, 5)
    windows = window_partition(x, 8)
    assert windows.shape == (3 * 2 * 3, 64, 5)


def test_each_window_holds_a_contiguous_patch():
    # Window 0 must be the top-left corner of the image, not a stride-y
    # scatter of pixels from across it.
    x = torch.arange(64, dtype=torch.float32).view(1, 8, 8, 1)
    first = window_partition(x, 4)[0].view(4, 4)
    torch.testing.assert_close(first, x[0, :4, :4, 0])


# --- attention -------------------------------------------------------------


def test_attention_preserves_shape():
    attention = WindowAttention(dim=16, num_heads=4, window=4)
    assert attention(torch.randn(6, 16, 16)).shape == (6, 16, 16)


def test_attention_rejects_heads_that_do_not_divide_the_width():
    with pytest.raises(ValueError, match="heads"):
        WindowAttention(dim=10, num_heads=4, window=4)


def test_attention_mixes_tokens_within_a_window():
    """The defining property, and the mechanism that can erase a lesion.

    Changing one token must change the others, because every token is a
    weighted blend of all of them. If it did not, this would be a pointwise
    layer wearing a transformer's name.
    """
    attention = WindowAttention(dim=16, num_heads=4, window=4).eval()
    a = torch.randn(1, 16, 16)
    b = a.clone()
    b[0, 0] += 5.0
    with torch.no_grad():
        out_a, out_b = attention(a), attention(b)
    # the token we did not touch still moved
    assert not torch.allclose(out_a[0, 7], out_b[0, 7], atol=1e-6)


def test_relative_position_bias_is_learnable():
    attention = WindowAttention(dim=16, num_heads=2, window=4)
    assert attention.relative_bias.requires_grad
    # one entry per (row offset, column offset) pair, per head
    assert attention.relative_bias.shape == ((2 * 4 - 1) ** 2, 2)


# --- blocks ----------------------------------------------------------------


@pytest.mark.parametrize("shift", [False, True])
def test_block_preserves_shape(shift):
    block = SwinBlock(dim=16, num_heads=4, window=4, shift=shift)
    assert block(torch.randn(2, 12, 12, 16)).shape == (2, 12, 12, 16)


def test_block_pads_sizes_the_window_does_not_divide():
    # 13 is not a multiple of 4; the block must pad internally and crop back
    # rather than refuse or silently truncate the image.
    block = SwinBlock(dim=8, num_heads=2, window=4)
    assert block(torch.randn(1, 13, 13, 8)).shape == (1, 13, 13, 8)


def test_shifted_block_builds_an_attention_mask():
    # Without the mask, the cyclic roll lets the bottom edge attend to the
    # top as though they were neighbours.
    shifted = SwinBlock(dim=8, num_heads=2, window=4, shift=True)
    mask = shifted._attention_mask(8, 8, torch.device("cpu"))
    assert mask is not None
    assert (mask < 0).any()  # some pairs are forbidden

    plain = SwinBlock(dim=8, num_heads=2, window=4, shift=False)
    assert plain._attention_mask(8, 8, torch.device("cpu")) is None


def test_shifting_changes_the_output():
    # The shift is the only mechanism by which information crosses a window
    # boundary. If it were a no-op, the model could never see past 8 pixels.
    torch.manual_seed(0)
    x = torch.randn(1, 16, 16, 8)
    plain = SwinBlock(dim=8, num_heads=2, window=8, shift=False).eval()
    shifted = SwinBlock(dim=8, num_heads=2, window=8, shift=True).eval()
    shifted.load_state_dict(plain.state_dict())
    with torch.no_grad():
        assert not torch.allclose(plain(x), shifted(x), atol=1e-6)


# --- resolution changes ----------------------------------------------------


def test_patch_merging_halves_resolution_and_doubles_width():
    merge = PatchMerging(dim=8)
    assert merge(torch.randn(2, 16, 16, 8)).shape == (2, 8, 8, 16)


def test_patch_expand_doubles_resolution_and_halves_width():
    expand = PatchExpand(dim=16)
    assert expand(torch.randn(2, 8, 8, 16)).shape == (2, 16, 16, 8)


def test_merge_then_expand_returns_to_the_original_shape():
    x = torch.randn(1, 16, 16, 8)
    assert PatchExpand(16)(PatchMerging(8)(x)).shape == x.shape


# --- the whole network -----------------------------------------------------


@pytest.mark.parametrize("size", [128, 64, 50, 96])
def test_output_is_one_value_per_input_pixel(size):
    net = SwinUNet(in_channels=6, embed_dim=16, depth=2, num_heads=2, window=8)
    assert net(torch.randn(1, 6, size, size)).shape == (1, size, size)


def test_handles_non_square_images():
    net = SwinUNet(in_channels=6, embed_dim=16, depth=2, num_heads=2)
    assert net(torch.randn(1, 6, 40, 56)).shape == (1, 40, 56)


def test_output_is_always_positive():
    # Same physical constraint as every other model here: T1 cannot be
    # negative, and the classical fit is bounded the same way.
    net = SwinUNet(in_channels=6, embed_dim=16, depth=1, num_heads=2)
    assert bool((net(torch.randn(2, 6, 32, 32) * 30.0) > 0).all())


def test_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="non-negative"):
        SwinUNet(depth=-1)
    with pytest.raises(ValueError, match="heads"):
        SwinUNet(embed_dim=10, num_heads=4)


def test_predict_ms_returns_milliseconds():
    net = SwinUNet(in_channels=6, embed_dim=16, depth=1, num_heads=2)
    out = net.predict_ms(torch.randn(1, 6, 32, 32))
    assert out.shape == (1, 32, 32)
    assert bool((out > 0).all())


def test_gradients_reach_every_parameter():
    # A disconnected branch would train silently and contribute nothing.
    net = SwinUNet(in_channels=6, embed_dim=16, depth=1, num_heads=2)
    net(torch.randn(1, 6, 32, 32)).sum().backward()
    missing = [n for n, p in net.named_parameters() if p.grad is None]
    assert not missing, f"no gradient reached: {missing}"


# --- the property the experiment rests on ---------------------------------


def test_attention_sees_further_than_convolution_at_equal_depth():
    """The reason a transformer is in this study at all.

    A convolution mixes a pixel with its immediate neighbours and gains
    reach only by stacking layers. Attention lets every pixel in a window
    reach every other in a single step. If spatial context is what erases
    small lesions, the wider-seeing architecture should erase more -- and
    that prediction is only meaningful if the view really is wider.
    """
    for depth in (1, 2):
        swin = SwinUNet(
            in_channels=6, embed_dim=16, depth=depth, num_heads=2, window=8
        )
        unet = UNet(in_channels=6, base_channels=8, depth=depth)
        assert receptive_field_px(swin, 256) > receptive_field_px(unet, 256)


def test_deeper_swin_sees_further():
    fields = [
        receptive_field_px(
            SwinUNet(in_channels=6, embed_dim=16, depth=d, num_heads=2), 256
        )
        for d in (0, 1, 2)
    ]
    assert fields == sorted(fields)
    assert len(set(fields)) == len(fields)
