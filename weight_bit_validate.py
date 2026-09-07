import argparse
import os
import time
import datetime

import yaml
import torch
import torch.nn.functional as F
from torch.cuda import amp
from torch.utils.data import DataLoader
from tqdm import tqdm
from spikingjelly.activation_based import encoding

from quant import (
    apply_channelwise_quantization,
    enable_qat_weight_fake_quant,
    enable_qat_activation_fake_quant,
    enable_activation_quant_inference,
)
from utils import *


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SNN Training & Quantization")
    parser.add_argument(
        "-config",
        type=str,
        default=None,
        help="path to YAML config file (CLI args override YAML values)",
    )
    parser.add_argument("-T", default=16, type=int, help="simulating time-steps")
    parser.add_argument("-device", default="cuda:0", help="device, e.g., cuda:0 or cpu")
    parser.add_argument("-b", default=16, type=int, help="batch size")
    parser.add_argument(
        "-epochs",
        default=64,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "-num_workers",
        default=4,
        type=int,
        metavar="N",
    )
    parser.add_argument(
        "-data-dir",
        type=str,
        default="/work/datasets",
        help="root dir of dataset (will be created/downloaded here)",
    )
    parser.add_argument(
        "-out-dir",
        type=str,
        default="./logs",
        help="root dir for saving logs and checkpoints",
    )
    parser.add_argument("-resume", type=str, help="resume from the checkpoint path")
    parser.add_argument(
        "-amp", action="store_true", help="automatic mixed precision training"
    )
    parser.add_argument("-cupy", action="store_true", help="use cupy backend")
    parser.add_argument(
        "-opt",
        type=str,
        default="adam",
        help="optimizer type: `sgd` or `adam`",
    )
    parser.add_argument(
        "-momentum", default=0.9, type=float, help="momentum for SGD optimizer"
    )
    parser.add_argument("-lr", default=0.001, type=float, help="learning rate")
    parser.add_argument(
        "-channels", default=128, type=int, help="channels of DVSGestureNet"
    )
    parser.add_argument(
        "-tau", default=2.0, type=float, help="membrane time constant of LIF neurons"
    )
    parser.add_argument(
        "-seed", default=42, type=int, help="random seed (PyTorch and CUDA)"
    )
    parser.add_argument(
        "-train",
        action="store_true",
        help="run training loop (default when --eval-only is not set)",
    )
    parser.add_argument(
        "-eval",
        action="store_true",
        help="only run evaluation with a pretrained checkpoint (no training)",
    )
    parser.add_argument(
        "-wandb",
        action="store_true",
        help="enable Weights & Biases (wandb) logging",
    )
    parser.add_argument(
        "-wandb-project",
        type=str,
        default="snn",
        help="wandb project name for logging",
    )
    parser.add_argument(
        "-wandb-name",
        type=str,
        default=None,
        help="wandb run name (optional)",
    )
    parser.add_argument(
        "-quant",
        action="store_true",
        help="enable quantization (applies to both training via QAT and evaluation)",
    )
    parser.add_argument(
        "-quant_target",
        type=str,
        default="weight",
        choices=["weight", "activation"],
        help="quantization target: 'weight' for channel-wise weight quantization, "
        "'activation' for LIF pre-activation quantization",
    )
    parser.add_argument(
        "-quant_bits",
        type=int,
        default=2,
        help="number of bits for quantization (2 or 4 for weight; arbitrary for activation)",
    )
    parser.add_argument(
        "-model",
        type=str,
        default='MNIST',
    )

    # Two-pass parsing: first extract -config, then apply YAML as defaults
    preliminary, _ = parser.parse_known_args()
    if preliminary.config is not None:
        with open(preliminary.config, "r") as f:
            yaml_cfg = yaml.safe_load(f)
        if yaml_cfg:
            parser.set_defaults(**yaml_cfg)
        print(f"Loaded config from {preliminary.config}")

    args = parser.parse_args()
    return args

def _prepare_batch(data, label, args, encoder=None):
    """
    Prepare a batch of data for the network, handling both DVS and static datasets.
    Returns (frame, label, label_onehot) ready for forward pass.
    """
    if args.model == 'MNIST':
        img = data.to(args.device)
        label = label.to(args.device)
        frame = torch.stack([encoder(img) for _ in range(args.T)], dim=0)
        label_onehot = F.one_hot(label, 10).float()
    elif args.model == 'DVSGesture':
        frame = data.to(args.device).transpose(0, 1)
        label = label.to(args.device)
        label_onehot = F.one_hot(label, 11).float()
    else:
        raise NotImplementedError(f"Unsupported model: {args.model}")
    return frame, label, label_onehot


def train_one_epoch(
    net: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: amp.GradScaler | None,
    args: argparse.Namespace,
    pbar: tqdm | None = None,
    epoch: int = 0,
) -> tuple[float, float, float]:
    """
    Train for one epoch and return (loss, accuracy, samples_per_second).
    """
    net.train()
    start_time = time.time()
    train_loss = 0.0
    train_acc = 0.0
    train_samples = 0

    encoder = encoding.PoissonEncoder() if args.model == 'MNIST' else None

    for data, label in train_loader:
        optimizer.zero_grad()
        frame, label, label_onehot = _prepare_batch(data, label, args, encoder)

        if scaler is not None:
            with amp.autocast():
                out_fr = net(frame).mean(0)
                loss = F.mse_loss(out_fr, label_onehot)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out_fr = net(frame).mean(0)
            loss = F.mse_loss(out_fr, label_onehot)
            loss.backward()
            optimizer.step()

        batch_size = label.numel()
        train_samples += batch_size
        train_loss += loss.item() * batch_size
        train_acc += (out_fr.argmax(1) == label).float().sum().item()

        functional.reset_net(net)

        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix(epoch=epoch, loss=loss.item())

    elapsed = time.time() - start_time
    train_speed = train_samples / max(elapsed, 1e-6)
    train_loss /= max(train_samples, 1)
    train_acc /= max(train_samples, 1)
    return train_loss, train_acc, train_speed

@torch.no_grad()
def evaluate(
    net: torch.nn.Module,
    test_loader: DataLoader,
    args: argparse.Namespace,
) -> tuple[float, float, float]:
    """
    Evaluate on test set and return (loss, accuracy, samples_per_second).
    """
    net.eval()

    start_time = time.time()
    test_loss = 0.0
    test_acc = 0.0
    test_samples = 0
    pbar = tqdm(test_loader, desc="Eval", leave=False)

    encoder = encoding.PoissonEncoder() if args.model == 'MNIST' else None

    for data, label in pbar:
        frame, label, label_onehot = _prepare_batch(data, label, args, encoder)

        out_fr = net(frame).mean(0)
        loss = F.mse_loss(out_fr, label_onehot)

        batch_size = label.numel()
        test_samples += batch_size
        test_loss += loss.item() * batch_size
        test_acc += (out_fr.argmax(1) == label).float().sum().item()

        functional.reset_net(net)
        pbar.set_postfix(loss=loss.item())

    elapsed = time.time() - start_time
    test_speed = test_samples / max(elapsed, 1e-6)
    test_loss /= max(test_samples, 1)
    test_acc /= max(test_samples, 1)

    # Print weight polarity statistics for the current network.
    total, num_pos, num_neg, num_zero, pos_ratio, neg_ratio, zero_ratio = summarize_weight_polarity(
        net
    )
    print(
        f"Weight polarity: total={total}, "
        f"pos={num_pos} ({pos_ratio:.4f}), "
        f"neg={num_neg} ({neg_ratio:.4f}), "
        f"zero={num_zero} ({zero_ratio:.4f})"
    )

    # 打印出网络中所有权重参数的最大值和最小值
    with torch.no_grad():
        weight_tensors = [
            p.data for p in net.parameters() if p.requires_grad and p.data.is_floating_point()
        ]
        if weight_tensors:
            all_weights = torch.cat([w.flatten() for w in weight_tensors])
            w_max = all_weights.max().item()
            w_min = all_weights.min().item()
            print(f"Weight max = {w_max:.6f}, min = {w_min:.6f}")
        else:
            print("No trainable floating-point weights found.")

    return test_loss, test_acc, test_speed


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)

    wandb = None
    if getattr(args, "wandb", False):
        try:
            import wandb as _wandb  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Failed to import wandb. Please install it (e.g. `pip install wandb`) "
                "or run without `-wandb`."
            ) from e
        wandb = _wandb

    net = build_model(args)

    train_loader, test_loader = build_dataloaders(args)
    optimizer, lr_scheduler = build_optimizer_and_scheduler(net, args)

    scaler: amp.GradScaler | None = amp.GradScaler() if args.amp else None

    start_epoch, max_test_acc = resume_if_needed(
        net, optimizer, lr_scheduler, args
    )

    # Set up quantization after checkpoint is loaded.
    # - Training: attach QAT hooks (fake quant with STE)
    # - Eval-only: apply real quantization (one-shot for weight, hooks for activation)
    if args.quant:
        if args.quant_target == "weight":
            if args.eval:
                apply_channelwise_quantization(net, num_bits=args.quant_bits)
            else:
                enable_qat_weight_fake_quant(net, num_bits=args.quant_bits)
        elif args.quant_target == "activation":
            if args.eval:
                enable_activation_quant_inference(net, num_bits=args.quant_bits)
            else:
                enable_qat_activation_fake_quant(net, num_bits=args.quant_bits)

    if args.model == 'MNIST':
        out_dir = os.path.join(
            args.out_dir,
            f"{args.model}_T{args.T}_b{args.b}_{args.opt}_lr{args.lr}",
        )
    else:
        out_dir = os.path.join(
            args.out_dir,
            f"{args.model}_T{args.T}_b{args.b}_{args.opt}_lr{args.lr}_c{args.channels}",
        )

    if args.amp:
        out_dir += "_amp"

    if args.cupy:
        out_dir += "_cupy"

    if args.quant:
        out_dir += f"_q{args.quant_target}_{args.quant_bits}bit"


    if not os.path.exists(out_dir) and args.train:
        os.makedirs(out_dir)
        print(f"Mkdir {out_dir}.")

    if wandb is not None:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=vars(args),
        )

    # Evaluation-only mode: load checkpoint and run a single evaluation, no training.
    if args.eval:
        if not args.resume:
            raise ValueError(
                "In --eval-only mode, please provide a pretrained checkpoint via `-resume`."
            )
        test_loss, test_acc, test_speed = evaluate(net, test_loader, args)
        if wandb is not None:
            wandb.log(
                {
                    "epoch": start_epoch,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                    "test_speed": test_speed,
                    "mode": "eval_only",
                }
            )
        print(
            f"[Eval only] test_loss = {test_loss:.4f}, "
            f"test_acc = {test_acc:.4f}, test_speed = {test_speed:.4f} images/s"
        )
        if wandb is not None:
            wandb.finish()
        return

    # Training (default when --eval-only is not set).
    total_steps = max(args.epochs - start_epoch, 0) * len(train_loader)
    train_pbar = tqdm(total=total_steps, desc="Train", unit="batch")

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        train_loss, train_acc, train_speed = train_one_epoch(
            net, train_loader, optimizer, scaler, args, pbar=train_pbar, epoch=epoch
        )
        lr_scheduler.step()

        test_loss, test_acc, test_speed = evaluate(net, test_loader, args)

        if wandb is not None:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                    "train_speed": train_speed,
                    "test_speed": test_speed,
                    "lr": lr_scheduler.get_last_lr()[0],
                },
                step=epoch,
            )

        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_max = True

        checkpoint = {
            "net": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "max_test_acc": max_test_acc,
        }

        if save_max:
            torch.save(checkpoint, os.path.join(out_dir, "checkpoint_max.pth"))

        torch.save(checkpoint, os.path.join(out_dir, "checkpoint_latest.pth"))

        print(out_dir)
        print(
            f"epoch = {epoch}, "
            f"train_loss = {train_loss:.4f}, train_acc = {train_acc:.4f}, "
            f"test_loss = {test_loss:.4f}, test_acc = {test_acc:.4f}, "
            f"max_test_acc = {max_test_acc:.4f}"
        )
        print(
            f"train speed = {train_speed:.4f} images/s, "
            f"test speed = {test_speed:.4f} images/s"
        )

        remaining_seconds = (time.time() - epoch_start) * (args.epochs - epoch - 1)
        escape_time = (
            datetime.datetime.now()
            + datetime.timedelta(seconds=remaining_seconds)
        ).strftime("%Y-%m-%d %H:%M:%S")
        print(f"estimated finish time = {escape_time}\n")

    train_pbar.close()
    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()

