import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from evaluation.metrics import min_ade, min_fde
from models.encoder import EncoderGRU
from models.decoder import MultiHeadDecoder

def evaluate_model(checkpoint_path, dataroot, hidden_dim):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running evaluation on {device}...")

    val_past = np.load(f"{dataroot}/val_past.npy")
    val_future = np.load(f"{dataroot}/val_future.npy")
    
    val_dataset = TensorDataset(torch.tensor(val_past, dtype=torch.float32), 
                                torch.tensor(val_future, dtype=torch.float32))
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Updated to match Member 1's architecture (input_dim=4, context_dim=hidden_dim*2)
    encoder = EncoderGRU(input_dim=4, hidden_dim=hidden_dim).to(device)
    decoder = MultiHeadDecoder(context_dim=hidden_dim * 2, hidden_dim=hidden_dim, pred_len=6, num_heads=3).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    
    print(f"Loaded checkpoint from Epoch {checkpoint.get('epoch', 'N/A')} "
          f"with Val Loss {checkpoint.get('val_loss', 'N/A')}")

    encoder.eval()
    decoder.eval()

    total_ade = 0.0
    total_fde = 0.0
    num_batches = 0

    with torch.no_grad():
        for past_traj, future_gt in val_loader:
            past_traj = past_traj.to(device)
            future_gt = future_gt.to(device)
            current_batch_size = past_traj.size(0)

            ego_hidden = encoder(past_traj)
            dummy_social_ctx = torch.zeros(current_batch_size, hidden_dim).to(device)
            combined_context = torch.cat([ego_hidden, dummy_social_ctx], dim=-1)
            
            predictions = decoder(combined_context) 

            batch_ade = min_ade(predictions, future_gt)
            batch_fde = min_fde(predictions, future_gt)

            total_ade += batch_ade
            total_fde += batch_fde
            num_batches += 1

    final_ade = total_ade / num_batches
    final_fde = total_fde / num_batches

    print("─────────────────────────────────────────")
    print("  Final Validation Results")
    print("─────────────────────────────────────────")
    print(f"  minADE@3  :  {final_ade:.4f} m")
    print(f"  minFDE@3  :  {final_fde:.4f} m")
    print(f"  Samples   :  {len(val_dataset)}")
    print("─────────────────────────────────────────")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--dataroot', type=str, default='data')
    parser.add_argument('--hidden_dim', type=int, default=64)
    args = parser.parse_args()
    
    evaluate_model(args.checkpoint, args.dataroot, args.hidden_dim)