import argparse
import os

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

from spikingjelly.activation_based import functional, surrogate, neuron
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture

from module import DVSGestureNet, MNISTSNN

def set_random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_model(args: argparse.Namespace) -> torch.nn.Module:
    """
    Build DVSGestureNet using spikingjelly's LIF neurons in multi-step mode.
    """
    if args.model == 'MNIST':
        net = MNISTSNN(
            tau = args.tau
        )
    elif args.model == 'DVSGesture':
        net = DVSGestureNet(
            channels=args.channels,
            spiking_neuron=neuron.LIFNode,
            surrogate_function=surrogate.ATan(),
            detach_reset=True,
            step_mode="m",
            backend="torch",
        )
    else:
        raise NotImplementedError(f"Unsupported model: {args.model}")
    functional.set_step_mode(net, "m")
    if args.cupy:
        functional.set_backend(net, "cupy", instance=neuron.LIFNode)
    net.to(args.device)
    return net


def build_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    os.makedirs(args.data_dir, exist_ok=True)

    if args.model == 'MNIST':
        mnist_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_set = torchvision.datasets.MNIST(
            root=args.data_dir,
            train=True,
            transform=mnist_transform,
            download=True,
        )
        test_set = torchvision.datasets.MNIST(
            root=args.data_dir,
            train=False,
            transform=mnist_transform,
            download=True,
        )
    elif args.model == 'DVSGesture':
        train_set = DVS128Gesture(
            root=args.data_dir,
            train=True,
            data_type="frame",
            frames_number=args.T,
            split_by="number",
        )
        test_set = DVS128Gesture(
            root=args.data_dir,
            train=False,
            data_type="frame",
            frames_number=args.T,
            split_by="number",
        )
    else:
        raise NotImplementedError(f"Unsupported model/dataset: {args.model}")

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.b,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        dataset=test_set,
        batch_size=args.b,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


def build_optimizer_and_scheduler(
    net: torch.nn.Module, args: argparse.Namespace
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """
    Build optimizer (SGD or Adam) and cosine annealing LR scheduler.
    """
    if args.opt.lower() == "sgd":
        optimizer = torch.optim.SGD(
            net.parameters(), lr=args.lr, momentum=args.momentum
        )
    elif args.opt.lower() == "adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    else:
        raise NotImplementedError(f"Unsupported optimizer type: {args.opt}")

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    return optimizer, lr_scheduler


def build_optimizer_and_scheduler_from_params(
    params: list,
    args: argparse.Namespace,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    if args.opt.lower() == "sgd":
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum)
    elif args.opt.lower() == "adam":
        optimizer = torch.optim.Adam(params, lr=args.lr)
    else:
        raise NotImplementedError(f"Unsupported optimizer type: {args.opt}")

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    return optimizer, lr_scheduler


def resume_if_needed(
    net: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    args: argparse.Namespace,
) -> tuple[int, float]:
    """
    Optionally resume training from a checkpoint.
    Returns:
        start_epoch, max_test_acc
    """
    start_epoch = 0
    max_test_acc = -1.0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        net.load_state_dict(checkpoint["net"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        max_test_acc = checkpoint["max_test_acc"]
        print(f"Resumed from {args.resume}, epoch={start_epoch - 1}")

    return start_epoch, max_test_acc

@torch.no_grad()
def summarize_weight_polarity(
    net: torch.nn.Module,
) -> tuple[int, int, int, int, float, float, float]:
    total = 0
    num_pos = 0
    num_neg = 0
    num_zero = 0

    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue
        data = param.data
        total += data.numel()
        num_pos += (data > 0).sum().item()
        num_neg += (data < 0).sum().item()
        num_zero += (data == 0).sum().item()

    if total == 0:
        return 0, 0, 0, 0, 0.0, 0.0, 0.0

    pos_ratio = num_pos / total
    neg_ratio = num_neg / total
    zero_ratio = num_zero / total
    return total, num_pos, num_neg, num_zero, pos_ratio, neg_ratio, zero_ratio


