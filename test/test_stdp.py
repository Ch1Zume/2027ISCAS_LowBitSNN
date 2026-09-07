import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import os
import torch
import torch.nn as nn

from snn import PoissonEncoder, LIFNeuron, LinearSNN
from visualize import plot_stdp_dynamics


class SimpleSTDPSNN(nn.Module):
    def __init__(self, in_features=28*28, out_features=100, T=20):
        super().__init__()
        self.T = T
        self.encoder = PoissonEncoder(T=T)
        self.syn = LinearSNN(
            in_features=in_features,
            out_features=out_features,
            stdp_enabled=True,
            tau_pre=0.9,
            tau_post=0.9,
            A_plus=0.01,
            A_minus=0.012,
            w_min=0.0,
            w_max=2.0,
        )
        self.neuron = LIFNeuron(
            tau=0.9,
            threshold=1.0,
            lens=0.5,
            reset_mode="subtract",
        )

    def forward(self, x):
        """
        x: [B, 1, 28, 28]
        """
        x = x.flatten(1)             # [B, 784]
        pre_spike = self.encoder(x)  # [T, B, 784]
        syn_current = self.syn(pre_spike)     # [T, B, out_features]
        post_spike = self.neuron(syn_current) # [T, B, out_features]
        return pre_spike, post_spike
    
    @torch.no_grad()
    def forward_from_spike(self, pre_spike):
        """
        pre_spike: [T, B, in_features]
        """
        syn_current = self.syn(pre_spike)         # [T, B, out_features]
        post_spike = self.neuron(syn_current)     # [T, B, out_features]
        return post_spike

    @torch.no_grad()
    def stdp_step(self, pre_spike, post_spike):
        self.syn.stdp_update(pre_spike, post_spike, reduction="mean")

    def reset_state(self):
        self.syn.reset_state()
        self.neuron.reset_state()

@torch.no_grad()
def run_random_stdp_visual_demo(
    T=100,
    spike_prob=0.3,
    weight_init=1.2,
    save_name="stdp_visual_random",
):
    """
    用随机 in_spike 驱动 1->1 SNN，
    out_spike 由 syn + LIFNeuron 自然产生，
    最后记录并可视化 STDP 动态。
    """
    net = SimpleSTDPSNN(in_features=1, out_features=1, T=T)

    net.syn.weight.data.fill_(weight_init)
    net.syn.w_max = max(net.syn.w_max, weight_init + 0.5)

    net.reset_state()
    net.syn.enable_recording()

    # 随机生成输入 spike: [T, B=1, N=1]
    pre_spike = (torch.rand(T, 1, 1) < spike_prob).float()

    # 通过 SNN 自然得到输出 spike
    post_spike = net.forward_from_spike(pre_spike)

    # 做 STDP 更新并记录 history
    net.stdp_step(pre_spike, post_spike)

    history = net.syn.get_history()

    plot_stdp_dynamics(
        history=history,
        batch_idx=0,
        in_idx=0,
        out_idx=0,
        title=f"{save_name}\nT={T}, spike_prob={spike_prob}, w0={weight_init}",
        save_path=os.path.join("test/figures", f"{save_name}.png"),
        show=True,
    )

'''
    验证STDP的更新方法是否已经融入了深度学习的生态
'''
def test_stdp_grad_equivalence():
    print("\n=== test_stdp_grad_equivalence ===")

    # 固定一组时序，里面同时包含 potentiation / depression 的成分
    pre_seq = torch.tensor([
        [[1.0]],
        [[0.0]],
        [[1.0]],
        [[0.0]],
        [[0.0]],
    ])  # [T, B=1, I=1]

    post_seq = torch.tensor([
        [[0.0]],
        [[1.0]],
        [[0.0]],
        [[1.0]],
        [[0.0]],
    ])  # [T, B=1, O=1]

    init_w = 1.0
    lr = 1.0
    scale = 1.0

    # -----------------------------
    # Path A: direct weight update
    # -----------------------------
    layer_direct = LinearSNN(
        in_features=1,
        out_features=1,
        stdp_enabled=True,
        A_plus=0.1,
        A_minus=0.1,
        tau_pre=0.9,
        tau_post=0.9,
        w_min=0.0,
        w_max=10.0,
        hard_clamp=False,   # 先关掉 clamp，避免比较时掺杂边界效应
    )
    layer_direct.weight.data.fill_(init_w)
    layer_direct.reset_state()

    layer_direct.stdp_update(
        pre_seq,
        post_seq,
        reduction="mean",
        on_grad=False,
        scale=scale,
    )

    w_direct = layer_direct.weight.detach().clone()

    # ----------------------------------------
    # Path B: pseudo-grad + SGD(momentum=0)
    # ----------------------------------------
    layer_grad = LinearSNN(
        in_features=1,
        out_features=1,
        stdp_enabled=True,
        A_plus=0.1,
        A_minus=0.1,
        tau_pre=0.9,
        tau_post=0.9,
        w_min=0.0,
        w_max=10.0,
        hard_clamp=False,
    )
    layer_grad.weight.data.fill_(init_w)
    layer_grad.reset_state()

    optimizer = torch.optim.SGD(layer_grad.parameters(), lr=lr, momentum=0.0)
    optimizer.zero_grad()

    layer_grad.stdp_update(
        pre_seq,
        post_seq,
        reduction="mean",
        on_grad=True,
        scale=scale,
    )

    print(f"pseudo grad = {layer_grad.weight.grad.item():.6f}")

    optimizer.step()

    w_grad = layer_grad.weight.detach().clone()

    print(f"w_direct = {w_direct.item():.6f}")
    print(f"w_grad   = {w_grad.item():.6f}")
    print(f"abs diff = {(w_direct - w_grad).abs().item():.6e}")

    assert torch.allclose(w_direct, w_grad, atol=1e-8), \
        "Direct STDP update and pseudo-grad SGD update are not equivalent."

    print("PASS")


if __name__ == "__main__":
    # run_random_stdp_visual_demo(
    #     T=150,
    #     spike_prob=0.3,
    #     weight_init=0.5,
    #     save_name="stdp_visual"
    # )
    test_stdp_grad_equivalence()