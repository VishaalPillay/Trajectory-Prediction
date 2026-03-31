import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from models.encoder import EncoderGRU
from models.decoder import MultiHeadDecoder

def plot_trajectories(checkpoint_path, dataroot, hidden_dim, n_samples):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('outputs/plots', exist_ok=True)

    val_past = np.load(f"{dataroot}/val_past.npy")
    val_future = np.load(f"{dataroot}/val_future.npy")

    # Match Member 1's architecture
    encoder = EncoderGRU(input_dim=4, hidden_dim=hidden_dim).to(device)
    decoder = MultiHeadDecoder(context_dim=hidden_dim * 2, hidden_dim=hidden_dim, pred_len=6, num_heads=3).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])

    encoder.eval()
    decoder.eval()
    colors = ['orange', 'green', 'red']
    
    for i in range(n_samples):
        past = torch.tensor(val_past[i], dtype=torch.float32).unsqueeze(0).to(device)
        gt_future = val_future[i]
        current_batch_size = 1

        with torch.no_grad():
            ego_hidden = encoder(past)
            dummy_social_ctx = torch.zeros(current_batch_size, hidden_dim).to(device)
            combined_context = torch.cat([ego_hidden, dummy_social_ctx], dim=-1)
            preds = decoder(combined_context).squeeze(0).cpu().numpy() 

        past_np = past.squeeze(0).cpu().numpy()
        
        # Grab the very last (x, y) point of the past trajectory to anchor the lines
        origin_x = past_np[-1, 0]
        origin_y = past_np[-1, 1]

        plt.figure(figsize=(8, 8))
        
        # Plot past track (extracting just x and y, ignoring dx, dy)
        plt.plot(past_np[:, 0], past_np[:, 1], 'b-o', label='Past Track', linewidth=2)
        plt.plot(origin_x, origin_y, 'bo', markersize=8) 

        # Prepend the origin to the Ground Truth to connect the line visually
        gt_x = np.concatenate([[origin_x], gt_future[:, 0]])
        gt_y = np.concatenate([[origin_y], gt_future[:, 1]])
        plt.plot(gt_x, gt_y, 'k--', alpha=0.5, label='Ground Truth', linewidth=2)

        # Prepend the origin to all 3 predictions
        for mode in range(3):
            pred_x = np.concatenate([[origin_x], preds[mode, :, 0]])
            pred_y = np.concatenate([[origin_y], preds[mode, :, 1]])
            plt.plot(pred_x, pred_y, color=colors[mode], marker='x', label=f'Pred Head {mode+1}')

        plt.title(f'Trajectory Prediction - Sample {i+1}')
        plt.xlabel('X Coordinate (m)')
        plt.ylabel('Y Coordinate (m)')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.axis('equal') 

        save_path = f'outputs/plots/sample_{i+1}.png'
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
    print(f"Saved {n_samples} plots to outputs/plots/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--dataroot', type=str, default='data')
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--n_samples', type=int, default=20)
    args = parser.parse_args()
    
    plot_trajectories(args.checkpoint, args.dataroot, args.hidden_dim, args.n_samples)