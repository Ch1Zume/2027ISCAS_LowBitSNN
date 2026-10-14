import torch
import torch.nn as nn
import torch.nn.functional as F
from .layer import LinearSNN


def train_with_stdp(model, train_loader, test_loader, args):
    device = args.device
    model.to(device)

    # -----------------------------
    # 参数分组
    # -----------------------------
    stdp_params = []
    stdp_param_ids = set()

    for m in model.modules():
        if isinstance(m, LinearSNN) and m.stdp_enabled:
            for p in m.parameters():
                stdp_params.append(p)
                stdp_param_ids.add(id(p))

    bp_params = []
    for p in model.parameters():
        if id(p) not in stdp_param_ids:
            bp_params.append(p)

    # -----------------------------
    # optimizer
    # -----------------------------
    optimizer_bp = torch.optim.Adam(
        bp_params,
        lr=args.lr_bp,
        weight_decay=args.weight_decay,
    )

    optimizer_stdp = torch.optim.SGD(
        stdp_params,
        lr=args.lr_stdp,
        momentum=0.0,
    )

    criterion = torch.nn.CrossEntropyLoss()

    # ======================================================
    # training loop
    # ======================================================
    for epoch in range(args.epochs):
        model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer_bp.zero_grad()
            optimizer_stdp.zero_grad()

            # --------------------------------
            # forward
            # 要求 model(images) 返回:
            # logits, stdp_ctx_list
            # stdp_ctx_list: list of dicts, each:
            #   {
            #       "module": LinearSNN (stdp_enabled=True),
            #       "pre_spike": Tensor,
            #       "post_spike": Tensor,
            #   }
            # --------------------------------
            logits, stdp_ctx_list = model(images)

            loss = criterion(logits, targets)

            # --------------------------------
            # BP backward
            # --------------------------------
            loss.backward()

            for module in model.modules():
                if isinstance(module, LinearSNN) and module.stdp_enabled:
                    if module.weight.grad is not None:
                        module.weight.grad.zero_()
                    if module.bias is not None and module.bias.grad is not None:
                        module.bias.grad.zero_()

            # --------------------------------
            # 对所有 STDP 层写入 STDP 伪梯度
            # --------------------------------
            for ctx in stdp_ctx_list:
                stdp_module = ctx["module"]
                pre_spike = ctx["pre_spike"]
                post_spike = ctx["post_spike"]

                stdp_module.stdp_update(
                    pre_spike=pre_spike,
                    post_spike=post_spike,
                    reduction="mean",
                    on_grad=True,
                    scale=args.stdp_scale,
                )

            # --------------------------------
            # 可选：只裁剪 BP 部分梯度
            # --------------------------------
            if hasattr(args, "grad_clip") and args.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(bp_params, args.grad_clip)

            # --------------------------------
            # optimizer update
            # --------------------------------
            optimizer_bp.step()
            optimizer_stdp.step()

            for module in model.modules():
                if isinstance(module, LinearSNN) and module.stdp_enabled and module.hard_clamp:
                    module.weight.data.clamp_(module.w_min, module.w_max)

            # --------------------------------
            # 统计
            # --------------------------------
            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size

            preds = logits.argmax(dim=1)
            total_correct += (preds == targets).sum().item()
            total_samples += batch_size

        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples

        # ======================================================
        # evaluation
        # ======================================================
        model.eval()

        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for images, targets in test_loader:
                images = images.to(device)
                targets = targets.to(device)

                logits, _ = model(images)
                loss = criterion(logits, targets)

                batch_size = targets.size(0)
                test_loss += loss.item() * batch_size

                preds = logits.argmax(dim=1)
                test_correct += (preds == targets).sum().item()
                test_total += batch_size

        test_loss = test_loss / test_total
        test_acc = test_correct / test_total

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} | "
            f"test_loss={test_loss:.4f} "
            f"test_acc={test_acc:.4f}"
        )

def eval_hybrid(model, test_loader, args):
    device = args.device
    model.to(device)
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            targets = targets.to(device)

            logits, _ = model(images)

            loss = criterion(logits, targets)

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size

            preds = logits.argmax(dim=1)
            total_correct += (preds == targets).sum().item()
            total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    print(f"[Eval] loss={avg_loss:.4f} acc={avg_acc:.4f}")

    return {
        "loss": avg_loss,
        "acc": avg_acc,
    }