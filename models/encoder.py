import torch
import torch.nn as nn

class EncoderGRU(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64):
        super().__init__()
        # We use a GRU as it has fewer parameters than an LSTM and trains faster
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        
    def forward(self, x):
        # x shape: (batch_size, T=4, 4)
        _, h_n = self.gru(x)
        
        # h_n shape: (1, batch_size, hidden_dim)
        # We squeeze out the layer dimension to get just (batch_size, hidden_dim)
        return h_n.squeeze(0)