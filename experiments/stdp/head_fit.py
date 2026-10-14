"""Nonnegative classifier fitting utility; independent of historical checkpoints."""

def fit_nonnegative_classifier(head, features, labels, teacher_targets, emit=None):
    """TRAIN-only ridge initialization and float BP, with a softmax-invariant gauge."""
    import torch
    from torch.nn import functional as F
    features, labels, teacher_targets = features.detach(), labels.detach(), teacher_targets.detach()
    mean = features.mean(0)
    centered = features-mean
    target = F.one_hot(labels, head.out_features).to(features)*5
    target_mean = target.mean(0)
    covariance = centered.T @ centered / len(features)
    covariance += torch.eye(head.in_features,device=features.device)*0.001
    coefficients = torch.linalg.solve(covariance, centered.T @ (target-target_mean)/len(features)).T
    with torch.no_grad():
        head.bias.copy_(target_mean-coefficients @ mean)
        head.weight.copy_(coefficients)
        # Adding the same feature coefficient to all classes shifts all logits
        # equally, preserving their differences while every physical weight is positive.
        head.weight.add_((0.25-head.weight.min(0).values).clamp_min(0).unsqueeze(0))
    optimizer = torch.optim.Adam(head.parameters(),lr=0.01)
    history = []
    for step in range(1,201):
        optimizer.zero_grad(set_to_none=True)
        logits = head(features)
        ce = F.cross_entropy(logits,labels)
        kd = F.kl_div(F.log_softmax(logits/2,dim=1),F.softmax(teacher_targets/2,dim=1),reduction='batchmean')*4
        loss = 0.5*ce+0.5*kd
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad.dtype == torch.float32 and torch.isfinite(p.grad).all() for p in head.parameters())
        optimizer.step()
        with torch.no_grad():
            head.weight.add_((0.25-head.weight.min(0).values).clamp_min(0).unsqueeze(0))
        assert head.weight.dtype == torch.float32 and head.weight.min() >= 0
        if step == 1 or step % 20 == 0:
            record = dict(step=step,loss=float(loss),ce=float(ce),kd=float(kd),
                          accuracy=float((logits.argmax(1)==labels).float().mean()))
            history.append(record)
            if emit is not None:
                emit('TRAIN_head_fit_progress',**record)
    return dict(partition='TRAIN',samples=len(features),steps=200,ridge=0.001,lr=0.01,
                targets='0.5 CE + 0.5 original TRAIN-only teacher KD, temperature 2',
                nonnegative_gauge_floor=0.25,history=history,hidden_and_backbone_frozen=True)
