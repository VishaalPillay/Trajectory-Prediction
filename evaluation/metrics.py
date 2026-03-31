import torch

def min_ade(predictions, ground_truth):
    # predictions: (batch_size, 3, pred_len, 2)
    # ground_truth: (batch_size, pred_len, 2)
    gt_expanded = ground_truth.unsqueeze(1)
    l2_distances = torch.norm(predictions - gt_expanded, dim=-1)
    ades = l2_distances.mean(dim=-1)
    min_ades, _ = torch.min(ades, dim=1)
    return min_ades.mean().item()

def min_fde(predictions, ground_truth):
    # Evaluate only the final timestep
    final_preds = predictions[:, :, -1, :] 
    final_gt = ground_truth[:, -1, :]      
    
    final_gt_expanded = final_gt.unsqueeze(1)
    final_distances = torch.norm(final_preds - final_gt_expanded, dim=-1)
    min_fdes, _ = torch.min(final_distances, dim=1)
    
    return min_fdes.mean().item()