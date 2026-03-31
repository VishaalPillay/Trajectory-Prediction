import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data.dataset import TrajectoryDataset
from models.encoder import EncoderGRU
from models.decoder import MultiHeadDecoder
from training.loss import combined_loss
import os

def train_model(epochs=30, batch_size=64, learning_rate=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")

    # 1. Setup DataLoaders
    # Note: paths assume we run from the project root.
    train_dataset = TrajectoryDataset('data/train_past.npy', 'data/train_future.npy', augment=True)
    val_dataset = TrajectoryDataset('data/val_past.npy', 'data/val_future.npy', augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Dataset Size: {len(train_dataset)} Train, {len(val_dataset)} Val")

    # 2. Setup Model Architecture
    hidden_dim = 64
    # Since social pooling is empty, we simulate it here just like the dummy pipeline.
    # We pass it to decoder.
    context_dim = hidden_dim * 2 

    # Inputs: (batch, T=4, dim=4: x,y,dx,dy)
    encoder = EncoderGRU(input_dim=4, hidden_dim=hidden_dim).to(device)
    # Output: (batch, num_modes=3, T=6, dim=2: x,y)
    decoder = MultiHeadDecoder(context_dim=context_dim, hidden_dim=hidden_dim, pred_len=6, num_heads=3).to(device)

    # 3. Setup Optimizer
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)

    best_val_loss = float('inf')

    # 4. Training Loop
    for epoch in range(epochs):
        encoder.train()
        decoder.train()
        total_train_loss = 0.0
        
        # WTA Alpha Warmup schedule (decays from 1.0 to 0.0 over the first 15 epochs)
        alpha = max(0.0, 1.0 - (epoch / 15.0))
        
        for batch_idx, (past, future) in enumerate(train_loader):
            past, future = past.to(device), future.to(device)
            current_batch_size = past.size(0)

            # --- FORWARD PASS ---
            optimizer.zero_grad()
            
            # Encode
            ego_hidden = encoder(past)
            
            # Dummy Social Context (Concatenate zeros for now as placeholder for social_pooling)
            dummy_social_ctx = torch.zeros(current_batch_size, hidden_dim).to(device)
            combined_context = torch.cat([ego_hidden, dummy_social_ctx], dim=-1)
            
            # Decode
            predictions = decoder(combined_context) # (batch, 3, 6, 2)
            
            # --- BACKWARD PASS ---
            loss = combined_loss(predictions, future, alpha=alpha)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # 5. Validation Loop
        encoder.eval()
        decoder.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for past, future in val_loader:
                past, future = past.to(device), future.to(device)
                current_batch_size = past.size(0)
                
                ego_hidden = encoder(past)
                dummy_social_ctx = torch.zeros(current_batch_size, hidden_dim).to(device)
                combined_context = torch.cat([ego_hidden, dummy_social_ctx], dim=-1)
                
                predictions = decoder(combined_context)
                
                # Validation loss with alpha=0 (pure Winner-Takes-All evaluation)
                val_loss = combined_loss(predictions, future, alpha=0.0)
                total_val_loss += val_loss.item()
                
        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f"Epoch [{epoch+1}/{epochs}] | Alpha (Warmup): {alpha:.2f} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"-> Validation loss decreased. Saving best.pt ...")
            torch.save({
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'epoch': epoch,
                'val_loss': best_val_loss,
            }, 'best.pt')

if __name__ == '__main__':
    # Launch training
    train_model(epochs=30)
