"""Tests for the networks and the shared training loop.

The important one here is ``test_mlp_learns_t1_from_clean_data``. Everything
else checks plumbing; that one checks the model can do the job at all, which
is the only reason to keep it in the comparison.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from qmri.dataset import generate_split, to_voxel_table
from qmri.models import VoxelMLP, to_ms, to_scaled
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
