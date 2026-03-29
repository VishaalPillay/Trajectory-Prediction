import torch

def combined_loss(pred, gt, alpha=0.0):
    """
    pred: Predicted trajectories (Batch, K=3, T=6, 2)
    gt: Ground truth trajectories (Batch, T=6, 2)
    alpha: Warmup weight (1.0 decays to 0.0)
    """
    # Expand ground truth to match the 3 prediction heads
    gt_exp = gt.unsqueeze(1).expand_as(pred) 
    
    # Calculate Euclidean distance across time and coordinates
    per_mode = torch.norm(pred - gt_exp, dim=-1).sum(-1) # (Batch, 3)

    # Winner-Takes-All (WTA) term: only penalize the head closest to ground truth
    best = per_mode.argmin(dim=1) # (Batch,)
    L_wta = per_mode.gather(1, best.unsqueeze(1)).squeeze(1).mean()

    # MSE warmup term: penalize all heads equally to prevent dead heads
    L_mse = per_mode.mean()

    return L_wta + alpha * L_mse