import random
import torch
from torch.utils.data import Dataset
from lln.data.pytorch.get_dataset import LongDataset

class PermutedLongDataset(Dataset):
    def __init__(self, original_dataset, feature_idx=None):
        self.original_dataset = original_dataset
        # Set the feature index to exchange
        self.feature_idx = feature_idx
        
    def __len__(self):
        return len(self.original_dataset.subjects)

    def __getitem__(self, idx):
        # Select the current subject
        subject_id = self.original_dataset.subjects[idx]
        subject_X = self.original_dataset.X[subject_id].clone()
        subject_y = self.original_dataset.y[subject_id].clone()
        
        # Randomly select another subject for feature exchange
        other_subject_id = random.choice([s for s in self.original_dataset.subjects if s != subject_id])
        
        # If feature_idx is not provided, raise an error
        if self.feature_idx is None:
            raise ValueError("You must provide a feature index to exchange.")
        
        # Ensure feature_idx is within the valid range
        assert 0 <= self.feature_idx < subject_X.shape[1], "Feature index out of bounds"

        # Swap the values for the given feature across all time points
        subject_X[:, self.feature_idx], self.original_dataset.X[other_subject_id][:, self.feature_idx] = \
            self.original_dataset.X[other_subject_id][:, self.feature_idx], subject_X[:, self.feature_idx]

        return subject_X, subject_y