import torch
from torch.utils.data import Dataset
import numpy as np
import random

class TrajectoryDataset(Dataset):
    def __init__(self, past_path, future_path, augment=False):
        """
        Loads .npy arrays (past/future trajectories) and applies
        preprocessing + data augmentation during training.
        """
        self.past_data = np.load(past_path)
        self.future_data = np.load(future_path)
        self.augment = augment

    def __len__(self):
        return len(self.past_data)

    def apply_augmentation(self, past, future):
        """
        Applies data augmentation to a single trajectory instance.
        """
        # 1. Random noise injection (prevents overfitting to exact paths)
        if random.random() < 0.5:
            # Add small gaussian noise to x,y positions
            noise_past = np.random.normal(0, 0.05, size=(past.shape[0], 2))
            past[:, :2] += noise_past
            
            noise_future = np.random.normal(0, 0.05, size=(future.shape[0], 2))
            future[:, :2] += noise_future

        # 2. Random scaling (simulates varying speeds/distances)
        if random.random() < 0.5:
            scale_factor = random.uniform(0.9, 1.1)
            past *= scale_factor
            future *= scale_factor

        return past, future

    def __getitem__(self, idx):
        past = self.past_data[idx].copy()
        future = self.future_data[idx].copy()

        if self.augment:
            past, future = self.apply_augmentation(past, future)

        # Convert to float32 tensors
        return torch.tensor(past, dtype=torch.float32), torch.tensor(future, dtype=torch.float32)
