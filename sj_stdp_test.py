"""
基于 SpikingJelly STDPLearner 的 MNIST 混合训练 pipeline.

架构:
  Poisson Encoder → Conv(1→16) → IF → Pool
                  → Conv(16→32) → IF → Pool
                  → Flatten → Linear(1568→256) → IF
                  → Linear(256→10)   ← 分类头, 无 IF

学习方式:
  - Conv 层: STDP (无监督特征学习)
  - Linear 层: BP  (有监督分类)
"""

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from spikingjelly.activation_based import learning, layer, neuron, functional


# ─────────────────────────── helpers ───────────────────────────


def poisson_encode(x: torch.Tensor, T: int) -> torch.Tensor:
    """Rate coding: 像素强度 → T 步泊松脉冲序列. x ∈ [0,1], out: [T,B,C,H,W]"""
    return (torch.rand(T, *x.shape, device=x.device) < x.unsqueeze(0)).float()


def f_weight(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, -1.0, 1.0)


# ──────────────────────────── model ────────────────────────────


def build_net():
    net = nn.Sequential(
        layer.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False),
        neuron.IFNode(),
        layer.MaxPool2d(2, 2),

        layer.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
        neuron.IFNode(),
        layer.MaxPool2d(2, 2),

        layer.Flatten(),

        layer.Linear(32 * 7 * 7, 256, bias=False),
        neuron.IFNode(),

        layer.Linear(256, 10, bias=False),
        # 分类头不加 IFNode，输出原始膜电位作 logits
    )
    functional.set_step_mode(net, "m")
    return net


# ──────────────────────── STDP wiring ──────────────────────────


STDP_LAYER_TYPES = (layer.Conv2d,)


def build_stdp_learners(net, tau_pre, tau_post):
    learners = []
    for i in range(len(net)):
        if isinstance(net[i], STDP_LAYER_TYPES):
            learners.append(
                learning.STDPLearner(
                    step_mode="m",
                    synapse=net[i],
                    sn=net[i + 1],
                    tau_pre=tau_pre,
                    tau_post=tau_post,
                    f_pre=f_weight,
                    f_post=f_weight,
                )
            )
    return learners


def split_params(net):
    """返回 (stdp_params, bp_params)."""
    stdp_ids = set()
    stdp_params = []
    for m in net.modules():
        if isinstance(m, STDP_LAYER_TYPES):
            for p in m.parameters():
                stdp_params.append(p)
                stdp_ids.add(id(p))

    bp_params = [p for p in net.parameters() if id(p) not in stdp_ids]
    return stdp_params, bp_params


# ─────────────────────── train / eval ──────────────────────────


def train_one_epoch(
    net, loader, optimizer_bp, optimizer_stdp, stdp_learners,
    device, T, epoch, stdp_scale=1.0,
):
    net.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(loader, desc=f"Train epoch {epoch}", leave=False)
    for img, label in pbar:
        img, label = img.to(device), label.to(device)

        # ── 编码 ──
        x_seq = poisson_encode(img, T)               # [T,B,1,28,28]

        # ── 重置 ──
        functional.reset_net(net)
        for lnr in stdp_learners:
            lnr.reset()

        # ── forward + BP loss ──
        optimizer_bp.zero_grad(set_to_none=True)
        optimizer_stdp.zero_grad(set_to_none=True)

        out_seq = net(x_seq)                          # [T,B,10]
        logits = out_seq.mean(0)                      # [B,10]
        loss = F.cross_entropy(logits, label)
        loss.backward()

        # ── STDP: 清除 BP 残留梯度, 写入 STDP 伪梯度 ──
        optimizer_stdp.zero_grad(set_to_none=True)
        for lnr in stdp_learners:
            lnr.step(on_grad=True, scale=stdp_scale)

        # ── step ──
        optimizer_bp.step()
        optimizer_stdp.step()

        # ── 统计 ──
        total_loss += loss.item() * label.size(0)
        pred = logits.argmax(1)
        total_correct += (pred == label).sum().item()
        total_samples += label.size(0)

        pbar.set_postfix(
            loss=f"{total_loss / total_samples:.4f}",
            acc=f"{total_correct / total_samples:.4f}",
        )

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(net, loader, device, T):
    net.eval()
    total_correct = 0
    total_samples = 0

    pbar = tqdm(loader, desc="Eval", leave=False)
    for img, label in pbar:
        img, label = img.to(device), label.to(device)
        x_seq = poisson_encode(img, T)

        functional.reset_net(net)
        out_seq = net(x_seq)
        logits = out_seq.mean(0)

        pred = logits.argmax(1)
        total_correct += (pred == label).sum().item()
        total_samples += label.size(0)
        pbar.set_postfix(acc=f"{total_correct / total_samples:.4f}")

    return total_correct / total_samples


# ──────────────────────── data loader ──────────────────────────


def build_dataloaders(data_dir, batch_size, num_workers):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, test_loader


# ────────────────────────── main ───────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="SpikingJelly STDP + BP MNIST Pipeline")
    p.add_argument("--T", type=int, default=10, help="SNN 时间步数")
    p.add_argument("--data-dir", type=str, default="./work/datasets")
    p.add_argument("-b", "--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr-bp", type=float, default=1e-3, help="BP 学习率")
    p.add_argument("--lr-stdp", type=float, default=1e-2, help="STDP 学习率")
    p.add_argument("--stdp-scale", type=float, default=1.0, help="STDP dW 缩放")
    p.add_argument("--tau-pre", type=float, default=2.0, help="STDP pre-trace 时间常数")
    p.add_argument("--tau-post", type=float, default=100.0, help="STDP post-trace 时间常数")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-path", type=str, default="best_sj_stdp.pth")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── data ──
    train_loader, test_loader = build_dataloaders(
        args.data_dir, args.batch_size, args.num_workers,
    )

    # ── model ──
    net = build_net().to(device)

    # ── STDP learners ──
    stdp_learners = build_stdp_learners(net, args.tau_pre, args.tau_post)

    # ── optimizers ──
    stdp_params, bp_params = split_params(net)
    optimizer_bp = Adam(bp_params, lr=args.lr_bp)
    optimizer_stdp = SGD(stdp_params, lr=args.lr_stdp, momentum=0.0)

    print(f"Device: {device} | T: {args.T} | Batch: {args.batch_size} | "
          f"Epochs: {args.epochs}")
    print(f"LR(bp): {args.lr_bp} | LR(stdp): {args.lr_stdp} | "
          f"Scale: {args.stdp_scale}")
    print(f"STDP layers: {len(stdp_learners)} conv | "
          f"BP params: {sum(p.numel() for p in bp_params)} | "
          f"STDP params: {sum(p.numel() for p in stdp_params)}")
    print("-" * 70)

    best_test_acc = 0.0

    for epoch in range(args.epochs):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            net, train_loader, optimizer_bp, optimizer_stdp, stdp_learners,
            device, args.T, epoch, stdp_scale=args.stdp_scale,
        )

        test_acc = evaluate(net, test_loader, device, args.T)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d} | "
              f"train loss {train_loss:.4f} | "
              f"train acc {train_acc:.4f} | "
              f"test acc {test_acc:.4f} | "
              f"time {elapsed:.1f}s")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(net.state_dict(), args.save_path)
            print(f"  -> best model saved to {args.save_path}")

    print("-" * 70)
    print(f"Best test acc: {best_test_acc:.4f}")


if __name__ == "__main__":
    main()
