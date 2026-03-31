import torch
import torch.nn as nn

class SimpleSocialPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # A simple linear network to mix the states of nearby agents
        self.spatial_mixer = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, ego_hidden):
        """
        Calculates social context by averaging the hidden states of 
        other agents in the current batch.
        """
        batch_size = ego_hidden.size(0)
        
        # If there's only one agent, there is no social context
        if batch_size == 1:
            return torch.zeros_like(ego_hidden)
            
        # Get the average state of all OTHER agents in the batch
        sum_hidden = torch.sum(ego_hidden, dim=0, keepdim=True)
        other_agents_sum = sum_hidden - ego_hidden 
        avg_social_state = other_agents_sum / (batch_size - 1)
        
        # Pass the crowd average through the linear mixer
        social_context = self.spatial_mixer(avg_social_state)
        
        return torch.relu(social_context)