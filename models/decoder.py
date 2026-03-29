import torch
import torch.nn as nn

class MultiHeadDecoder(nn.Module):
    def __init__(self, context_dim=128, hidden_dim=64, pred_len=6, num_heads=3):
        super().__init__()
        self.pred_len = pred_len
        self.num_heads = num_heads
        
        # Project the concatenated [ego + social] context back to hidden_dim
        self.context_proj = nn.Linear(context_dim, hidden_dim)
        
        # Create independent GRU heads and output layers
        self.heads = nn.ModuleList([
            nn.GRU(input_size=2, hidden_size=hidden_dim, batch_first=True) 
            for _ in range(num_heads)
        ])
        
        self.out = nn.ModuleList([
            nn.Linear(hidden_dim, 2) 
            for _ in range(num_heads)
        ])

    def forward(self, context):
        # context shape: (batch_size, context_dim)
        batch_size = context.size(0)
        
        # Project context and prepare for GRU initialization: (1, batch_size, hidden_dim)
        h_init = self.context_proj(context).unsqueeze(0)
        
        all_trajs = []
        
        for head, out_layer in zip(self.heads, self.out):
            h_t = h_init
            
            # Start token for autoregressive decoding (zeros)
            inp = torch.zeros(batch_size, 1, 2, device=context.device)
            preds = []
            
            for _ in range(self.pred_len):
                out, h_t = head(inp, h_t)
                step = out_layer(out.squeeze(1)) # shape: (batch_size, 2)
                preds.append(step)
                inp = step.unsqueeze(1) # Feed prediction as next input
                
            # Stack predictions along the time dimension
            all_trajs.append(torch.stack(preds, dim=1)) # (batch_size, T, 2)
            
        # Stack all 3 modes together
        return torch.stack(all_trajs, dim=1) # (batch_size, 3, T, 2)