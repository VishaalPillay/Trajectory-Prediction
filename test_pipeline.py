import torch
import torch.optim as optim
from models.encoder import EncoderGRU
from models.decoder import MultiHeadDecoder
from training.loss import combined_loss

def run_dummy_test():
    batch_size = 16
    hidden_dim = 64
    # For this test, we assume context_dim is 2*hidden_dim (to simulate concatenated ego + social context)
    context_dim = hidden_dim * 2 

    # Initialize modules
    encoder = EncoderGRU(input_dim=4, hidden_dim=hidden_dim)
    decoder = MultiHeadDecoder(context_dim=context_dim, hidden_dim=hidden_dim, pred_len=6, num_heads=3)
    
    # Setup optimizer
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)

    print("Generating dummy tensors...")
    # Dummy past trajectory: 4 timesteps of (x, y, dx, dy)
    past_traj = torch.randn(batch_size, 4, 4) 
    # Dummy future ground truth: 6 timesteps of (x, y)
    future_gt = torch.randn(batch_size, 6, 2) 
    
    # --- FORWARD PASS ---
    print("Running forward pass...")
    # 1. Encode past
    ego_hidden = encoder(past_traj)
    
    # 2. Simulate social pooling by just concatenating random noise of the same shape
    dummy_social_ctx = torch.randn(batch_size, hidden_dim)
    combined_context = torch.cat([ego_hidden, dummy_social_ctx], dim=-1)
    
    # 3. Decode into 3 distinct trajectories
    predictions = decoder(combined_context)
    
    print(f"Predictions shape: {predictions.shape} (Expected: {batch_size}, 3, 6, 2)")
    
    # --- BACKWARD PASS ---
    print("Running backward pass...")
    optimizer.zero_grad()
    
    # Compute loss with warmup active (alpha=1.0)
    loss = combined_loss(predictions, future_gt, alpha=1.0)
    loss.backward()
    optimizer.step()
    
    print(f"Success! Pipeline ran without crashing. Final Loss: {loss.item():.4f}")

if __name__ == "__main__":
    run_dummy_test()