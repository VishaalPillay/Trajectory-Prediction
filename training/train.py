import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os

from models.encoder import EncoderGRU
from models.decoder import MultiHeadDecoder
from models.social_pooling import SimpleSocialPooling
from training.loss import combined_loss

def train_model(dataroot, epochs, warmup_epochs, hidden_dim, batch_size, lr, save_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting training on {device}...")

    # 1. Load Preprocessed Data
    train_past = np.load(f"{dataroot}/train_past.npy")
    train_future = np.load(f"{dataroot}/train_future.npy")
    val_past = np.load(f"{dataroot}/val_past.npy")
    val_future = np.load(f"{dataroot}/val_future.npy")

    train_dataset = TensorDataset(torch.tensor(train_past, dtype=torch.float32), 
                                  torch.tensor(train_future, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(val_past, dtype=torch.float32), 
                                torch.tensor(val_future, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 2. Initialize the Complete Architecture
    encoder = EncoderGRU(input_dim=4, hidden_dim=hidden_dim).to(device)
    pooling = SimpleSocialPooling(hidden_dim=hidden_dim).to(device)
    decoder = MultiHeadDecoder(context_dim=hidden_dim * 2, hidden_dim=hidden_dim, pred_len=6, num_heads=3).to(device)

    # We must optimize all three components
    optimizer = optim.Adam(list(encoder.parameters()) + 
                           list(pooling.parameters()) + 
                           list(decoder.parameters()), lr=lr)

    best_val_loss = float('inf')

    # 3. Training Loop
    for epoch in range(1, epochs + 1):
        encoder.train()
        pooling.train()
        decoder.train()

        # Calculate alpha for warmup (decays from 1.0 to 0.0)
        alpha = max(0.0, 1.0 - (epoch - 1) / warmup_epochs) if warmup_epochs > 0 else 0.0

        total_train_loss = 0.0

        for past_traj, future_gt in train_loader:
            past_traj = past_traj.to(device)
            future_gt = future_gt.to(device)

            optimizer.zero_grad()

            # --- Full Forward Pass ---
            ego_hidden = encoder(past_traj)
            social_ctx = pooling(ego_hidden)  # Actual social context!
            
            combined_context = torch.cat([ego_hidden, social_ctx], dim=-1)
            predictions = decoder(combined_context)

            # Compute Loss and Optimize
            loss = combined_loss(predictions, future_gt, alpha)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # 4. Validation Loop
        encoder.eval()
        pooling.eval()
        decoder.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for past_traj, future_gt in val_loader:
                past_traj = past_traj.to(device)
                future_gt = future_gt.to(device)

                ego_hidden = encoder(past_traj)
                social_ctx = pooling(ego_hidden)
                combined_context = torch.cat([ego_hidden, social_ctx], dim=-1)

                predictions = decoder(combined_context)
                
                # Validation always uses alpha=0.0 (Pure Winner-Takes-All)
                loss = combined_loss(predictions, future_gt, alpha=0.0) 
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        print(f"Epoch [{epoch}/{epochs}] | Alpha: {alpha:.2f} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # 5. Save the Best Checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print("-> Validation loss decreased. Saving best.pt ...")
            torch.save({
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'epoch': epoch,
                'val_loss': best_val_loss,
            }, os.path.join(save_dir, 'best.pt'))

    print("Training complete! Model saved to best.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='data')
    parser.add_argument('--epochs', type=int, default=50) # Set to 5-10 if you are out of time!
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save_dir', type=str, default='.')
    args = parser.parse_args()
    
    train_model(args.dataroot, args.epochs, args.warmup, args.hidden, args.batch, args.lr, args.save_dir)