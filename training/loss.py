import torch

def combined_loss(predictions, ground_truth, alpha):
    """
    predictions: (batch, 3, pred_len, 2)
    ground_truth: (batch, pred_len, 2)
    alpha: warmup weight that decays from 1.0 to 0.0 over time.
    """
    # Expand ground truth to match the 3 prediction heads
    gt_expanded = ground_truth.unsqueeze(1) 

    # Calculate L2 distance (error) for every coordinate step
    l2_distances = torch.norm(predictions - gt_expanded, dim=-1) 

    # Sum the errors over the 6 timesteps for each head
    trajectory_errors = l2_distances.sum(dim=-1) 

    # 1. Winner-Takes-All (WTA) Loss: Find the head closest to the ground truth
    min_errors, _ = torch.min(trajectory_errors, dim=1)
    loss_wta = min_errors.mean()

    # 2. MSE Warmup Loss: Penalize all heads equally at the start of training
    loss_mse = trajectory_errors.mean()

    # Combine them using the alpha decay
    total_loss = loss_wta + (alpha * loss_mse)
    
    return total_loss