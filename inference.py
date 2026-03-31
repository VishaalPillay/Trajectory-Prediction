import torch
import numpy as np
import json
import os
import argparse
from models.encoder import EncoderGRU
from models.decoder import MultiHeadDecoder

def generate_submission(checkpoint_path, dataroot, hidden_dim, output_file):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Generating predictions on {device}...")

    # Load the evaluation/test data
    val_past = np.load(f"{dataroot}/val_past.npy")
    
    # Initialize models
    encoder = EncoderGRU(input_dim=4, hidden_dim=hidden_dim).to(device)
    decoder = MultiHeadDecoder(context_dim=hidden_dim * 2, hidden_dim=hidden_dim, pred_len=6, num_heads=3).to(device)

    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    
    encoder.eval()
    decoder.eval()

    submission_dict = {}

    print("Running forward pass...")
    with torch.no_grad():
        for i in range(len(val_past)):
            # Prepare tensor
            past = torch.tensor(val_past[i], dtype=torch.float32).unsqueeze(0).to(device)
            current_batch_size = 1
            
            # Forward pass
            ego_hidden = encoder(past)
            # Use zeros if social pooling isn't ready, otherwise use the pooling layer
            dummy_social_ctx = torch.zeros(current_batch_size, hidden_dim).to(device)
            combined_context = torch.cat([ego_hidden, dummy_social_ctx], dim=-1)
            
            # Predictions shape: (1, 3, 6, 2) -> (3, 6, 2)
            preds = decoder(combined_context).squeeze(0).cpu().numpy()
            
            # Anchor predictions to the last known position (to prevent teleporting)
            origin_x = val_past[i][-1, 0]
            origin_y = val_past[i][-1, 1]
            
            anchored_preds = []
            for mode in range(3):
                mode_path = []
                for step in range(6):
                    # Add origin to offsets to get absolute coordinates
                    abs_x = float(preds[mode, step, 0] + origin_x)
                    abs_y = float(preds[mode, step, 1] + origin_y)
                    mode_path.append([abs_x, abs_y])
                anchored_preds.append(mode_path)
            
            # Create a mock token ID since we don't have the raw strings
            sample_id = f"agent_sample_{i:05d}"
            submission_dict[sample_id] = anchored_preds

    # Ensure submission directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(submission_dict, f, indent=2)
        
    print(f"Successfully saved {len(val_past)} predictions to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--dataroot', type=str, default='data')
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--output', type=str, default='submission/predictions.json')
    args = parser.parse_args()
    
    generate_submission(args.checkpoint, args.dataroot, args.hidden_dim, args.output)