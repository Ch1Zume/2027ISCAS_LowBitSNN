import argparse
import time

import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

from snn import (
    Conv2dSNN,
    LinearSNN,
    FlattenSNN,
    MaxPool2dSNN,
    BatchNorm2dSNN,
    LIFNeuron,
    RepeatEncoder,
    PoissonEncoder
)
from utils import build_dataloaders, build_optimizer_and_scheduler_from_params


class MNISTSNN(nn.Module):

    def __init__(self, T: int, stdp: bool = False):
        super().__init__()

        self.encoder = PoissonEncoder(T)
        self.stdp = stdp

        # ── conv backbone ──
        self.conv1 = Conv2dSNN(1, 32, kernel_size=3, padding=1)
        self.bn1 = BatchNorm2dSNN(32)
        self.lif1 = LIFNeuron()
        self.pool1 = MaxPool2dSNN(2)

        self.conv2 = Conv2dSNN(32, 64, kernel_size=3, padding=1)
        self.bn2 = BatchNorm2dSNN(64)
        self.lif2 = LIFNeuron()
        self.pool2 = MaxPool2dSNN(2)

        self.flatten = FlattenSNN()

        # ── fc layers ──
        self.fc1 = LinearSNN(64 * 7 * 7, 256, stdp_enabled=stdp)
        self.lif_fc1 = LIFNeuron()

        # fc2: 分类头始终用 BP，不加 LIF — 直接输出膜电位作 logits
        self.fc2 = LinearSNN(256, 10, stdp_enabled=False)

        self._fc_pairs = [
            (self.fc1, self.lif_fc1),
            (self.fc2, None),
        ]

    def forward(self, x):
        # x : [B,1,28,28]
        x = self.encoder(x)      # [T,B,1,28,28]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.lif1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.lif2(x)
        x = self.pool2(x)

        x = self.flatten(x)      # [T,B,N]

        stdp_ctx_list = []

        for idx, (fc, lif) in enumerate(self._fc_pairs):
            pre_spike = x
            x = fc(x)
            if lif is not None:
                x = lif(x)
            post_spike = x

            if fc.stdp_enabled:
                stdp_ctx_list.append({
                    "module": fc,
                    "pre_spike": pre_spike.detach(),
                    "post_spike": post_spike.detach(),
                })

        # temporal average readout
        logits = x.mean(0)       # [B,10]

        return logits, stdp_ctx_list


def reset_net(net):

    for m in net.modules():
        if hasattr(m, "reset_state"):
            m.reset_state()


def get_amp_context(amp_mode, device):
    """Return (autocast context manager, GradScaler or None)."""
    if amp_mode == "fp16":
        return (
            torch.amp.autocast(device.type, dtype=torch.float16),
            torch.amp.GradScaler(device.type),
        )
    elif amp_mode == "bf16":
        return (
            torch.amp.autocast(device.type, dtype=torch.bfloat16),
            None,
        )
    else:
        return torch.amp.autocast(device.type, enabled=False), None


def _fmt(t: torch.Tensor) -> str:
    return (f"mean={t.mean().item():.6f} std={t.std().item():.6f} "
            f"min={t.min().item():.6f} max={t.max().item():.6f}")


def train_one_epoch(
    net, loader, optimizer_bp, device, autocast_ctx, scaler, epoch,
    optimizer_stdp=None, stdp_scale=1.0, grad_clip=None,
    debug_batches=3,
):
    net.train()

    total_loss = 0
    total_correct = 0
    total_samples = 0
    has_stdp = optimizer_stdp is not None

    pbar = tqdm(loader, desc=f"Train epoch {epoch}", leave=False)
    for batch_idx, (img, label) in enumerate(pbar):

        img = img.to(device)
        label = label.to(device)
        do_debug = (debug_batches > 0) and (batch_idx < debug_batches) and (epoch == 0)

        reset_net(net)

        optimizer_bp.zero_grad(set_to_none=True)
        if has_stdp:
            optimizer_stdp.zero_grad(set_to_none=True)

        with autocast_ctx:
            out, stdp_ctx_list = net(img)
            loss = F.cross_entropy(out, label)

        if do_debug:
            print(f"\n[DEBUG batch={batch_idx}] loss={loss.item():.4f}")
            print(f"  logits: {_fmt(out)}")

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if has_stdp:
            if do_debug:
                for name, m in net.named_modules():
                    if isinstance(m, LinearSNN) and m.stdp_enabled:
                        bp_grad = m.weight.grad
                        if bp_grad is not None:
                            print(f"  {name} BP grad:  {_fmt(bp_grad)}")

            for m in net.modules():
                if isinstance(m, LinearSNN) and m.stdp_enabled:
                    if m.weight.grad is not None:
                        m.weight.grad.zero_()
                    if m.bias is not None and m.bias.grad is not None:
                        m.bias.grad.zero_()

            for ctx in stdp_ctx_list:
                module = ctx["module"]
                pre_spike = ctx["pre_spike"]
                post_spike = ctx["post_spike"]

                if do_debug:
                    name = [n for n, m in net.named_modules() if m is module][0]
                    print(f"  {name} pre_spike:  {_fmt(pre_spike)} "
                          f"fire_rate={pre_spike.mean().item():.4f}")
                    print(f"  {name} post_spike: {_fmt(post_spike)} "
                          f"fire_rate={post_spike.mean().item():.4f}")

                module.reset_state()

                module.stdp_update(
                    pre_spike=pre_spike,
                    post_spike=post_spike,
                    reduction="mean",
                    on_grad=True,
                    scale=stdp_scale,
                )

                if do_debug:
                    print(f"  {name} pseudo_grad: {_fmt(module.weight.grad)}")
                    print(f"  {name} weight (before step): {_fmt(module.weight.data)}")

        if grad_clip is not None:
            if scaler is not None:
                scaler.unscale_(optimizer_bp)
            torch.nn.utils.clip_grad_norm_(optimizer_bp.param_groups[0]["params"], grad_clip)

        if scaler is not None:
            scaler.step(optimizer_bp)
            if has_stdp:
                scaler.step(optimizer_stdp)
            scaler.update()
        else:
            optimizer_bp.step()
            if has_stdp:
                optimizer_stdp.step()

        if has_stdp:
            for m in net.modules():
                if isinstance(m, LinearSNN) and m.stdp_enabled:
                    m.normalize_weights()

        if do_debug:
            for name, m in net.named_modules():
                if isinstance(m, LinearSNN) and m.stdp_enabled:
                    print(f"  {name} weight (after norm):  {_fmt(m.weight.data)}")

        total_loss += loss.item() * label.size(0)
        pred = out.argmax(1)
        total_correct += (pred == label).sum().item()
        total_samples += label.size(0)

        pbar.set_postfix(loss=f"{total_loss / total_samples:.4f}",
                         acc=f"{total_correct / total_samples:.4f}")

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(net, loader, device, autocast_ctx):

    net.eval()

    total_correct = 0
    total_samples = 0

    pbar = tqdm(loader, desc="Eval", leave=False)
    for img, label in pbar:

        img = img.to(device)
        label = label.to(device)

        reset_net(net)

        with autocast_ctx:
            out, _ = net(img)

        pred = out.argmax(1)
        total_correct += (pred == label).sum().item()
        total_samples += label.size(0)

        pbar.set_postfix(acc=f"{total_correct / total_samples:.4f}")

    return total_correct / total_samples

def parse_args():
    parser = argparse.ArgumentParser(description="SNN MNIST Training / Eval")

    # --- mode ---
    subparsers = parser.add_subparsers(dest="mode", required=True,
                                       help="train or eval")

    # ── train ──
    train_p = subparsers.add_parser("train", help="train the model")
    train_p.add_argument("--T", type=int, default=4, help="SNN time steps")
    train_p.add_argument("--data-dir", type=str, default="./data")
    train_p.add_argument("-b", "--batch-size", type=int, default=128, dest="b")
    train_p.add_argument("--num-workers", type=int, default=4)
    train_p.add_argument("--opt", type=str, default="adam",
                         choices=["sgd", "adam"])
    train_p.add_argument("--lr", type=float, default=1e-3)
    train_p.add_argument("--momentum", type=float, default=0.9)
    train_p.add_argument("--epochs", type=int, default=30)
    train_p.add_argument("--amp", type=str, default="fp32",
                         choices=["fp32", "fp16", "bf16"])
    train_p.add_argument("--seed", type=int, default=42)
    train_p.add_argument("--save-path", type=str, default="best_snn.pth",
                         help="path to save best model checkpoint")
    train_p.add_argument("--resume", type=str, default=None,
                         help="checkpoint path to resume training from")
    train_p.add_argument("--grad-clip", type=float, default=None,
                         help="max norm for gradient clipping (BP params only)")
    train_p.add_argument("--stdp", action="store_true",
                         help="enable STDP for all Linear layers")
    train_p.add_argument("--lr-stdp", type=float, default=1.0,
                         help="learning rate for STDP optimizer (SGD)")
    train_p.add_argument("--stdp-scale", type=float, default=1.0,
                         help="scaling factor for STDP pseudo-gradient")

    # ── eval ──
    eval_p = subparsers.add_parser("eval", help="evaluate a saved checkpoint")
    eval_p.add_argument("--T", type=int, default=4, help="SNN time steps")
    eval_p.add_argument("--data-dir", type=str, default="./data")
    eval_p.add_argument("-b", "--batch-size", type=int, default=128, dest="b")
    eval_p.add_argument("--num-workers", type=int, default=4)
    eval_p.add_argument("--amp", type=str, default="fp32",
                         choices=["fp32", "fp16", "bf16"])
    eval_p.add_argument("--seed", type=int, default=42)
    eval_p.add_argument("--checkpoint", type=str, required=True,
                         help="path to model checkpoint for evaluation")

    return parser.parse_args()


def setup_device_and_amp(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.amp != "fp32" and device.type == "cpu":
        print(f"[WARN] AMP mode '{args.amp}' requested but running on CPU, "
              "falling back to fp32.")
        args.amp = "fp32"

    if args.amp == "bf16" and device.type == "cuda":
        if not torch.cuda.is_bf16_supported():
            print("[WARN] BF16 not supported on this GPU, falling back to fp16.")
            args.amp = "fp16"

    return device


def do_train(args):

    device = setup_device_and_amp(args)

    args.model = "MNIST"
    train_loader, test_loader = build_dataloaders(args)

    net = MNISTSNN(args.T, stdp=args.stdp).to(device)

    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=True)
        net.load_state_dict(state)
        print(f"Resumed weights from {args.resume}")

    # ── 参数分组 ──
    stdp_param_ids = set()
    stdp_params = []
    for m in net.modules():
        if isinstance(m, LinearSNN) and m.stdp_enabled:
            for p in m.parameters():
                stdp_params.append(p)
                stdp_param_ids.add(id(p))

    bp_params = [p for p in net.parameters() if id(p) not in stdp_param_ids]

    # ── optimizers ──
    optimizer_bp, scheduler = build_optimizer_and_scheduler_from_params(
        bp_params, args,
    )

    optimizer_stdp = None
    if stdp_params:
        optimizer_stdp = torch.optim.SGD(stdp_params, lr=args.lr_stdp, momentum=0.0)

    autocast_ctx, scaler = get_amp_context(args.amp, device)

    mode_str = "BP+STDP" if stdp_params else "BP"
    print(f"Device: {device} | Mode: {mode_str} | Precision: {args.amp} | "
          f"Batch: {args.b} | LR(bp): {args.lr} | Epochs: {args.epochs}")
    if stdp_params:
        print(f"LR(stdp): {args.lr_stdp} | Scale: {args.stdp_scale}")
    print("-" * 70)

    best_test_acc = 0.0

    for epoch in range(args.epochs):

        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            net, train_loader, optimizer_bp, device, autocast_ctx, scaler, epoch,
            optimizer_stdp=optimizer_stdp,
            stdp_scale=args.stdp_scale,
            grad_clip=args.grad_clip,
        )

        test_acc = evaluate(net, test_loader, device, autocast_ctx)

        scheduler.step()

        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:3d} | "
            f"train loss {train_loss:.4f} | "
            f"train acc {train_acc:.4f} | "
            f"test acc {test_acc:.4f} | "
            f"time {elapsed:.1f}s"
        )

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            if args.save_path:
                torch.save(net.state_dict(), args.save_path)
                print(f"  -> best model saved to {args.save_path}")

    print("-" * 70)
    print(f"Best test acc: {best_test_acc:.4f}")


def do_eval(args):

    device = setup_device_and_amp(args)

    args.model = "MNIST"
    _, test_loader = build_dataloaders(args)

    net = MNISTSNN(args.T).to(device)

    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    net.load_state_dict(state)
    print(f"Loaded checkpoint from {args.checkpoint}")

    autocast_ctx, _ = get_amp_context(args.amp, device)

    test_acc = evaluate(net, test_loader, device, autocast_ctx)

    print("-" * 70)
    print(f"Test acc: {test_acc:.4f}")


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "train":
        do_train(args)
    else:
        do_eval(args)