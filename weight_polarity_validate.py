# mnist_neg_weight_ablation_spikingjelly.py
# pip install spikingjelly torch torchvision

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

from spikingjelly.activation_based import layer, neuron, surrogate, encoding, functional


# ----------------------------
# Utils
# ----------------------------
def clamp_nonneg_weights_(model: nn.Module):
    for m in model.modules():
        if hasattr(m, "weight") and (m.weight is not None):
            m.weight.data.clamp_(0.0)

def weight_report(model: nn.Module, title: str):
    print(f"\n== {title} weight report ==")
    for name, m in model.named_modules():
        if hasattr(m, "weight") and (m.weight is not None):
            w = m.weight.data
            neg = int((w < 0).sum().item())
            print(f"{name:35s}  min={w.min().item():+.4f}  max={w.max().item():+.4f}  #neg={neg}")


# ----------------------------
# Model: very small CNN-SNN
# ----------------------------
class SmallCNNSNN(nn.Module):
    """
    A tiny CNN SNN:
      Conv -> LIF -> AvgPool
      Conv -> LIF -> AvgPool
      Flatten -> Linear -> LIF -> Linear
    We run T steps outside, average logits.
    """
    def __init__(self, in_ch=1, base=16, tau=2.0):
        super().__init__()
        self.conv1 = layer.Conv2d(in_ch, base, kernel_size=3, padding=1, bias=False)
        self.lif1 = neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan())

        self.conv2 = layer.Conv2d(base, base * 2, kernel_size=3, padding=1, bias=False)
        self.lif2 = neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan())

        self.pool = nn.AvgPool2d(2)  # 28->14->7

        self.fc1 = layer.Linear((base * 2) * 7 * 7, 128, bias=True)
        self.lif3 = neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan())
        self.fc2 = layer.Linear(128, 10, bias=True)

    def forward(self, x_t):
        # x_t: [B, C, 28, 28], values in {0,1} if using PoissonEncoder
        x = self.conv1(x_t)
        x = self.lif1(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.lif2(x)
        x = self.pool(x)

        x = x.flatten(1)
        x = self.fc1(x)
        x = self.lif3(x)
        x = self.fc2(x)
        return x  # logits [B,10]


# ----------------------------
# Train / Eval
# ----------------------------
def build_input(x, input_mode: str):
    # x in [0,1]
    if input_mode == "raw1":
        return x
    elif input_mode == "onoff2":
        # [x, 1-x]  -> channel doubles
        return torch.cat([x, 1.0 - x], dim=1)
    else:
        raise ValueError(input_mode)

@torch.no_grad()
def evaluate(model, loader, encoder, T, device, input_mode):
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        x_in = build_input(x, input_mode)

        logits_sum = 0.0
        for _ in range(T):
            spk = encoder(x_in)  # Poisson spikes
            logits_sum += model(spk)
        logits = logits_sum / T

        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()

        functional.reset_net(model)
    return correct / total

def train(model, train_loader, test_loader, encoder, T, device, input_mode,
          nonneg: bool, epochs=5, lr=1e-3, wd=0.0):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            x_in = build_input(x, input_mode)

            opt.zero_grad()

            logits_sum = 0.0
            for _ in range(T):
                spk = encoder(x_in)
                logits_sum += model(spk)
            logits = logits_sum / T

            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()

            if nonneg:
                clamp_nonneg_weights_(model)

            with torch.no_grad():
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.numel()
                total_loss += loss.item() * y.numel()

            functional.reset_net(model)

        train_acc = correct / total
        test_acc = evaluate(model, test_loader, encoder, T, device, input_mode)
        print(f"Epoch {ep:02d}: loss={total_loss/total:.4f} | train_acc={train_acc:.4f} | test_acc={test_acc:.4f}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    # MNIST
    tfm = transforms.Compose([
        transforms.ToTensor(),  # [0,1]
    ])
    root = os.path.expanduser("/work/datasets/")
    train_set = datasets.MNIST(root, train=True, download=True, transform=tfm)
    test_set  = datasets.MNIST(root, train=False, download=True, transform=tfm)

    # For a quick demo, you can reduce data:
    # train_set = torch.utils.data.Subset(train_set, range(20000))
    # test_set  = torch.utils.data.Subset(test_set, range(5000))

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    # Encoder: Poisson rate coding
    encoder = encoding.PoissonEncoder()

    T = 16
    epochs = 5

    # A) Signed weights, raw input
    print("\n=== Model A: Signed weights (raw1) ===")
    model_A = SmallCNNSNN(in_ch=1).to(device)
    train(model_A, train_loader, test_loader, encoder, T, device, input_mode="raw1",
          nonneg=False, epochs=epochs, lr=1e-3)
    acc_A = evaluate(model_A, test_loader, encoder, T, device, "raw1")
    weight_report(model_A, "Model A (Signed, raw1)")
    print(f"Final test acc A = {acc_A:.4f}")

    # B) Non-negative weights, raw input
    print("\n=== Model B: NonNeg weights (raw1) ===")
    model_B = SmallCNNSNN(in_ch=1).to(device)
    clamp_nonneg_weights_(model_B)
    train(model_B, train_loader, test_loader, encoder, T, device, input_mode="raw1",
          nonneg=True, epochs=epochs, lr=1e-3)
    acc_B = evaluate(model_B, test_loader, encoder, T, device, "raw1")
    weight_report(model_B, "Model B (NonNeg, raw1)")
    print(f"Final test acc B = {acc_B:.4f}")

    # C) Non-negative weights, ON/OFF input channels
    print("\n=== Model C: NonNeg weights (onoff2) ===")
    model_C = SmallCNNSNN(in_ch=2).to(device)
    clamp_nonneg_weights_(model_C)
    train(model_C, train_loader, test_loader, encoder, T, device, input_mode="onoff2",
          nonneg=True, epochs=epochs, lr=1e-3)
    acc_C = evaluate(model_C, test_loader, encoder, T, device, "onoff2")
    weight_report(model_C, "Model C (NonNeg, onoff2)")
    print(f"Final test acc C = {acc_C:.4f}")

    print("\nSummary:")
    print(f"A (Signed, raw1)   test_acc = {acc_A:.4f}")
    print(f"B (NonNeg, raw1)   test_acc = {acc_B:.4f}")
    print(f"C (NonNeg, onoff2) test_acc = {acc_C:.4f}")


if __name__ == "__main__":
    main()
