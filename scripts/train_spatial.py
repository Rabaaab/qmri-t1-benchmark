"""Train a spatial model and score it against every other method.

Run with:

    python scripts/train_spatial.py --arch unet --depth 3

Writes ``outputs/<arch>_d<depth>.pt`` and prints the full comparison.

A spatial model is the first in this project that can see a pixel's
neighbours. Everything before it -- the classical fit, and the MLP --
worked one pixel at a time and therefore could not smooth anything, which
is why they all returned flat lesion retention regardless of size.

Two things keep the comparison honest. Inputs are scale-normalised exactly
as the MLP's were, and the loss is masked to tissue, because the voxel table
dropped background before the MLP ever saw it. Without the mask a spatial
model would bank an easy win on the ~46% of each image that is air.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from qmri.dataset import Dataset, to_image_tensors
from qmri.evaluate import summarise
from qmri.models import UNet, receptive_field_px, to_scaled
from qmri.swin import SwinUNet
from qmri.tissues import LESION_3T, TISSUES_3T
from qmri.training import TrainConfig, train

TISSUE_NAMES = {t.label: t.name for t in TISSUES_3T if t.t1_ms > 0}
TISSUE_NAMES[LESION_3T.label] = LESION_3T.name

ARCHITECTURES = ("unet", "swin")


def build_model(args, in_channels: int):
    """One place where each architecture's own arguments are applied.

    The two take different shape parameters -- channel width for the
    convolutional net, token width and attention heads for the transformer
    -- but `depth` means the same thing in both: how many times the image
    is halved, and therefore how far the model can see.
    """
    if args.arch == "unet":
        return UNet(
            in_channels=in_channels,
            base_channels=args.base_channels,
            depth=args.depth,
        )
    return SwinUNet(
        in_channels=in_channels,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        window=args.window,
        patch_size=args.patch_size,
    )


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--arch", default="unet", choices=ARCHITECTURES)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--base-channels", type=int, default=32, help="unet only")
    p.add_argument("--embed-dim", type=int, default=32, help="swin only")
    p.add_argument("--num-heads", type=int, default=4, help="swin only")
    p.add_argument("--window", type=int, default=8, help="swin only")
    p.add_argument("--patch-size", type=int, default=2, help="swin only")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--device",
        default="cpu",
        help='"cuda" on a GPU machine such as Colab; "cpu" otherwise',
    )
    p.add_argument("--data", type=Path, default=ROOT / "data")
    p.add_argument("--out", type=Path, default=ROOT / "outputs")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    for name in ("train", "val", "test"):
        if not (args.data / f"{name}.npz").exists():
            print(f"missing {args.data / f'{name}.npz'}; run build_dataset.py")
            return 1

    train_set = Dataset.load(args.data / "train.npz")
    val_set = Dataset.load(args.data / "val.npz")
    test_set = Dataset.load(args.data / "test.npz")

    train_x, train_y, train_mask = to_image_tensors(train_set)
    val_x, val_y, val_mask = to_image_tensors(val_set)
    test_x, _, _ = to_image_tensors(test_set)

    def factory():
        return build_model(args, in_channels=train_x.shape[1])

    probe = factory()
    field = receptive_field_px(probe, size=256)
    n_params = sum(p.numel() for p in probe.parameters())

    print(f"{args.arch}, depth {args.depth}, {n_params:,} parameters")
    print(f"receptive field {field} px, against lesions 2 to 12 px across")
    print(f"training on {len(train_set)} images, {train_x.shape[1]} channels\n")

    started = time.perf_counter()
    model, history = train(
        factory,
        train_x,
        to_scaled(train_y),
        val_x,
        to_scaled(val_y),
        TrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            patience=args.patience,
            seed=args.seed,
            device=args.device,
        ),
        train_mask=train_mask,
        val_mask=val_mask,
    )
    minutes = (time.perf_counter() - started) / 60.0
    print(
        f"\ntrained in {minutes:.1f} min, best epoch "
        f"{history.best_epoch + 1} of {history.epochs_run}, "
        f"val loss {history.best_val_loss:.6f}"
    )

    tag = f"{args.arch}_d{args.depth}"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": vars(args),
            "receptive_field_px": field,
            "n_params": n_params,
            "best_val_loss": history.best_val_loss,
        },
        args.out / f"{tag}.pt",
    )

    # --- predict the test split, one image at a time to stay in memory ---
    model.eval()
    predicted = np.empty_like(test_set.t1)
    with torch.no_grad():
        for i in range(len(test_set)):
            # The model may be on a GPU, so the batch has to follow it there
            # and the result has to come back before numpy will touch it.
            batch = torch.as_tensor(
                test_x[i : i + 1], dtype=torch.float32, device=args.device
            )
            predicted[i] = model.predict_ms(batch).cpu().numpy()[0]

    # --- gather every method we have ------------------------------------
    methods: list[tuple[str, np.ndarray]] = []
    baseline = args.data / "baseline_test.npz"
    if baseline.exists():
        with np.load(baseline) as f:
            methods.append(("fit, no B1", f["naive"]))
            methods.append(("fit, with B1", f["corrected"]))

    mlp_path = args.out / "mlp.pt"
    if mlp_path.exists():
        from qmri.dataset import scatter_to_maps, to_voxel_table
        from qmri.models import VoxelMLP

        table_x, _, table_mask = to_voxel_table(test_set)
        checkpoint = torch.load(mlp_path, weights_only=False)
        mlp = VoxelMLP(
            n_features=table_x.shape[1],
            hidden=checkpoint["config"].get("hidden", 256),
            depth=checkpoint["config"].get("depth", 3),
        )
        mlp.load_state_dict(checkpoint["state_dict"])
        with torch.no_grad():
            flat = mlp.predict_ms(torch.as_tensor(table_x)).numpy()
        methods.append(("MLP", scatter_to_maps(flat, table_mask)))

    methods.append((f"{args.arch} d{args.depth}", predicted))

    results = {name: summarise(est, test_set) for name, est in methods}

    # --- per-tissue bias and scatter -------------------------------------
    print()
    header = f"{'tissue':<14}" + "".join(f"{n:>22}" for n, _ in methods)
    print(header)
    print("-" * len(header))
    for label, tissue in TISSUE_NAMES.items():
        row = f"{tissue:<14}"
        for name, _ in methods:
            stats = results[name]["tissues"].get(label)
            row += (
                f"{stats.bias:>14.1f} ±{stats.sd:<6.1f}" if stats else f"{'-':>22}"
            )
        print(row)
    print("\nbias ± scatter, in ms")

    # --- the two headline numbers ----------------------------------------
    print()
    print(f"{'':<14}{'whole-image RMSE':>20}{'lesion RMSE':>14}{'retention':>18}")
    print("-" * 66)
    for name, _ in methods:
        r = results[name]
        print(
            f"{name:<14}{r['rmse_whole']:>20.1f}{r['rmse_lesions']:>14.1f}"
            f"{r['retention_mean']:>12.3f} ±{r['retention_sd']:<5.3f}"
        )

    # --- retention against lesion size: the headline ---------------------
    print()
    print(f"{'lesion size (px)':<18}{'n':>5}" + "".join(f"{n:>20}" for n, _ in methods))
    print("-" * (23 + 20 * len(methods)))
    size_rows = {name: results[name]["retention_by_size"] for name, _ in methods}
    for i, (label, count, _, _) in enumerate(size_rows[methods[0][0]]):
        row = f"{label:<18}{count:>5}"
        for name, _ in methods:
            _, _, mean, sd = size_rows[name][i]
            row += f"{mean:>13.2f} ±{sd:<5.2f}"
        print(row)

    print()
    print(
        "Retention 1.0 means the lesion came through intact; below 1 means it"
    )
    print(
        "faded. A downward trend toward small sizes is the effect under study."
    )
    print(f"\nwrote {(args.out / f'{tag}.pt').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
