"""Tests for the networks and the shared training loop.

The important one here is ``test_mlp_learns_t1_from_clean_data``. Everything
else checks plumbing; that one checks the model can do the job at all, which
is the only reason to keep it in the comparison.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from qmri.dataset import generate_split, to_voxel_table
from qmri.models import UNet, VoxelMLP, receptive_field_px, to_ms, to_scaled
from qmri.training import TrainConfig, train


def tiny_table(n=6, shape=(40, 40), seed=0, snr=1e9):
    ds = generate_split(n, shape, np.random.SeedSequence(seed), snr=snr)
    x, y, _ = to_voxel_table(ds)
    return x, to_scaled(y)


# --- scaling ---------------------------------------------------------------


def test_scaling_round_trips():
    assert to_ms(to_scaled(832.0)) == pytest.approx(832.0)


def test_targets_are_order_one_after_scaling():
    # Squared error on numbers near 1000 gives gradients near a million,
    # which is a needless fight with the optimiser.
    _, y = tiny_table()
    assert 0.1 < float(np.abs(y).mean()) < 10.0


# --- the model -------------------------------------------------------------


def test_forward_shape_is_one_value_per_voxel():
    model = VoxelMLP(n_features=6)
    assert model(torch.randn(17, 6)).shape == (17,)


def test_output_is_always_positive():
    # T1 cannot be negative. The classical fit is bounded the same way, so
    # neither method may return a physically impossible value.
    model = VoxelMLP(n_features=6)
    wild = torch.randn(512, 6) * 50.0
    assert bool((model(wild) > 0).all())


def test_depth_changes_parameter_count():
    shallow = sum(p.numel() for p in VoxelMLP(depth=1).parameters())
    deep = sum(p.numel() for p in VoxelMLP(depth=4).parameters())
    assert deep > shallow


def test_rejects_zero_depth():
    with pytest.raises(ValueError, match="at least 1"):
        VoxelMLP(depth=0)


def test_predict_ms_returns_milliseconds():
    model = VoxelMLP(n_features=6)
    out = model.predict_ms(torch.randn(8, 6))
    assert out.shape == (8,)
    # softplus output times the scale, so strictly positive milliseconds
    assert bool((out > 0).all())


# --- the capability test ---------------------------------------------------


def test_mlp_learns_t1_from_clean_data():
    """A per-voxel network can recover T1, so it earns its place as control.

    Noiseless data with the true B1 supplied, which is the easiest version
    of the task: the classical fit solves it to machine precision. The MLP
    will not match that, but it must get close enough that any later failure
    is attributable to noise or to context, not to the model being unable to
    represent the mapping.
    """
    train_x, train_y = tiny_table(n=8, seed=0)
    val_x, val_y = tiny_table(n=3, seed=1)

    model, history = train(
        lambda: VoxelMLP(n_features=train_x.shape[1], hidden=128, depth=3),
        train_x,
        train_y,
        val_x,
        val_y,
        TrainConfig(epochs=120, batch_size=2048, patience=25),
        verbose=False,
    )

    predicted = model.predict_ms(torch.as_tensor(val_x)).numpy()
    truth = to_ms(val_y)
    median_error = float(np.median(np.abs(predicted - truth) / truth))

    assert history.best_val_loss < 0.05
    assert median_error < 0.10


# --- the training loop -----------------------------------------------------


def test_training_reduces_loss():
    train_x, train_y = tiny_table(n=4)
    val_x, val_y = tiny_table(n=2, seed=1)
    _, history = train(
        lambda: VoxelMLP(n_features=train_x.shape[1], hidden=64, depth=2),
        train_x,
        train_y,
        val_x,
        val_y,
        TrainConfig(epochs=15, patience=99),
        verbose=False,
    )

    assert history.train_loss[-1] < history.train_loss[0]
    assert history.epochs_run == 15


def test_early_stopping_restores_the_best_weights():
    """The reported model must be the best one, not the last one.

    Trained with a deliberately huge learning rate so validation loss
    diverges after an early minimum; the weights left loaded must be the
    ones from the best epoch.
    """
    train_x, train_y = tiny_table(n=4)
    val_x, val_y = tiny_table(n=2, seed=1)
    model, history = train(
        lambda: VoxelMLP(n_features=train_x.shape[1], hidden=64, depth=2),
        train_x,
        train_y,
        val_x,
        val_y,
        TrainConfig(epochs=40, learning_rate=0.5, patience=4),
        verbose=False,
    )

    with torch.no_grad():
        loaded = float(
            torch.nn.functional.mse_loss(
                model(torch.as_tensor(val_x)), torch.as_tensor(val_y)
            )
        )
    assert loaded == pytest.approx(history.best_val_loss, rel=1e-5)
    assert history.best_epoch < history.epochs_run - 1


def test_same_seed_gives_the_same_model():
    train_x, train_y = tiny_table(n=4)
    val_x, val_y = tiny_table(n=2, seed=1)

    def run():
        # A factory, so the seed covers weight initialisation too. Passing a
        # pre-built model would make this test fail: its weights would come
        # from whatever RNG state the previous run left behind.
        _, h = train(
            lambda: VoxelMLP(n_features=train_x.shape[1], hidden=32, depth=2),
            train_x,
            train_y,
            val_x,
            val_y,
            TrainConfig(epochs=6, seed=7, patience=99),
            verbose=False,
        )
        return h.val_loss

    np.testing.assert_allclose(run(), run(), rtol=1e-10)


def test_history_records_the_best_epoch():
    train_x, train_y = tiny_table(n=3)
    val_x, val_y = tiny_table(n=2, seed=1)
    _, history = train(
        lambda: VoxelMLP(n_features=train_x.shape[1], hidden=32, depth=2),
        train_x,
        train_y,
        val_x,
        val_y,
        TrainConfig(epochs=10, patience=99),
        verbose=False,
    )

    assert 0 <= history.best_epoch < history.epochs_run
    assert history.best_val_loss == min(history.val_loss)


# --- the U-Net -------------------------------------------------------------


def test_unet_returns_one_map_per_image():
    net = UNet(in_channels=6, base_channels=8, depth=2)
    assert net(torch.randn(3, 6, 64, 64)).shape == (3, 64, 64)


def test_unet_output_is_always_positive():
    # Same physical constraint as the MLP, so neither model is allowed a
    # value the classical fit could not produce.
    net = UNet(in_channels=6, base_channels=8, depth=2)
    assert bool((net(torch.randn(2, 6, 32, 32) * 40.0) > 0).all())


def test_unet_handles_sizes_that_are_not_powers_of_two():
    # 50 does not divide by 4, so the decoder must pad rather than crash.
    net = UNet(in_channels=6, base_channels=8, depth=2)
    assert net(torch.randn(1, 6, 50, 50)).shape == (1, 50, 50)


def test_unet_handles_non_square_images():
    net = UNet(in_channels=6, base_channels=8, depth=2)
    assert net(torch.randn(1, 6, 40, 56)).shape == (1, 40, 56)


def test_unet_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="non-negative"):
        UNet(depth=-1)
    with pytest.raises(ValueError, match="at least 1"):
        UNet(base_channels=0)


def test_unet_predict_ms_returns_milliseconds():
    net = UNet(in_channels=6, base_channels=8, depth=1)
    out = net.predict_ms(torch.randn(1, 6, 32, 32))
    assert out.shape == (1, 32, 32)
    assert bool((out > 0).all())


# --- receptive field: the number that makes the ablation mean something ----


def test_receptive_field_is_never_zero_for_a_real_network():
    """Regression test.

    An all-zero probe leaves every ReLU exactly at its kink, where the
    gradient is defined as zero, so nothing propagates and the measurement
    silently returns 0. Probes must be random.
    """
    net = UNet(in_channels=6, base_channels=8, depth=4)
    assert receptive_field_px(net, size=256) > 1


def test_receptive_field_grows_with_depth():
    """Depth is the independent variable, and this is what it controls.

    Each level halves the image, so a deeper network draws on a wider
    region -- which is the distance over which it is able to smooth a
    lesion away.
    """
    fields = [
        receptive_field_px(UNet(in_channels=6, base_channels=8, depth=d), 256)
        for d in (0, 1, 2, 3)
    ]
    assert fields == sorted(fields)
    assert len(set(fields)) == len(fields)  # strictly increasing


def test_shallowest_network_is_almost_local():
    # depth 0 has no downsampling at all: two 3x3 convolutions, so it sees
    # five pixels across. It is the extra control at the local end.
    net = UNet(in_channels=6, base_channels=8, depth=0)
    assert receptive_field_px(net, size=64) == 5


def test_receptive_field_warns_when_clipped_by_the_probe():
    # A silently clipped value would understate how far a deep network can
    # smooth, and quietly weaken the ablation.
    net = UNet(in_channels=6, base_channels=8, depth=4)
    with pytest.warns(UserWarning, match="lower bound"):
        receptive_field_px(net, size=32)


def test_receptive_field_is_reproducible():
    net = UNet(in_channels=6, base_channels=8, depth=2)
    assert receptive_field_px(net, 128, seed=3) == receptive_field_px(
        net, 128, seed=3
    )


def test_receptive_field_leaves_training_mode_untouched():
    # It flips the model to eval to stop BatchNorm mixing the batch; it must
    # put it back, or a later training run silently behaves differently.
    net = UNet(in_channels=6, base_channels=8, depth=1)
    net.train()
    receptive_field_px(net, 64)
    assert net.training
