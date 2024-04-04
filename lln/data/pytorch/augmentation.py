"""Augment time series data.
"""
from torch.utils.data import Dataset
import numpy as np
import torch
import sys

class AugmentedDataset(Dataset):
    '''Dataset for producing augmentations'''	
    def __init__(self, dataset, augment_operations, random_transforms):
        self.y_dtype = dataset.y_dtype
        self.random_transforms = random_transforms
        # Augment operations are static operations that are applied to the data
        X = dataset.X
        y = dataset.y
        subjects = dataset.subjects
        for function_name, params in augment_operations:
            X, y, subjects = globals()[function_name](X, y, subjects, **params)
        self.X, self.y, self.subjects = X, y, subjects
    
    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subject_id = self.subjects[idx]
        X = self.X[subject_id]
        y = self.y[subject_id]
        # Random transforms are applied when fetching the item
        for function_name, params in self.random_transforms:
            X, y = globals()[function_name](X, y, **params)
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=self.y_dtype)
        return X, y

def zero_after_ix(X, y, ix=2):
    '''Replace all elements after ix with 0s.'''
    X[ix+1:] = np.zeros_like(X[0])
    y[ix+1:] = 0
    return X, y

def pad(X, y, subjects, max_length=None):
    '''Pad the sequences to the same length'''
    if max_length is None:
        max_length = max([len(X[subject]) for subject in subjects])
    for subject in subjects:
        X[subject] = np.pad(X[subject], ((0, max_length-len(X[subject])), (0, 0)), mode='constant')
        y[subject] = np.pad(y[subject], (0, max_length-len(y[subject])), mode='constant')
    return X, y, subjects

# Static augmentation operations
def upsample(X, y, subjects, per_time_point=True, reverse=True):
    '''Upsample the minority class. Note that reverse==True gives more heterogeneity.'''
    assert per_time_point, "Only per time point upsampling is supported"
    new_X, new_y, new_subjects = X.copy(), y.copy(), subjects.copy()
    timepoints = range(len(y[subjects[0]]))
    if reverse:
        timepoints = reversed(timepoints)
    for t_ix in timepoints:
        #print("Balancing timepoint", t_ix, "Total subjects", len(new_subjects))
        y_t = np.array([new_y[subject][t_ix] for subject in new_subjects])
        unique, counts = np.unique(y_t, return_counts=True)
        #print("Counts before", counts)
        for class_ix, count in zip(unique, counts):
            if count < max(counts):
                #print("Upsampling class", class_ix, count, max(counts))
                minority_subjects = [s for s in new_subjects if new_y[s][t_ix] == class_ix]
                duplicated_subjects = np.random.choice(minority_subjects, size=max(counts)-count)
                for new_subject_ix, old_subject in enumerate(duplicated_subjects):
                    new_subject = f"upsampled_{t_ix}_{class_ix}_{new_subject_ix}"
                    new_subjects.append(new_subject)    
                    new_X[new_subject], new_y[new_subject] = new_X[old_subject], new_y[old_subject]
        y_t = np.array([new_y[subject][t_ix] for subject in new_subjects])
        unique, counts = np.unique(y_t, return_counts=True)
        #print("Counts after", unique, counts)
    return new_X, new_y, new_subjects

# Random augmentation operations
def mask_sequence(X, y, mask_fraction=0.1):
    '''Mask a fraction of the data'''
    # Randomly mask a fraction of the data
    y_mask = np.random.choice([0, 1], size=y.shape, p=[mask_fraction, 1-mask_fraction])
    X_mask = np.expand_dims(y_mask, axis=-1)
    X_mask = np.repeat(X_mask, X.shape[1], axis=1)
    y = y * y_mask
    X = X * X_mask
    return X, y

def perturb_data(X, y, x_noise=0.1, y_prob=0.1, nr_classes=3):
    '''Add some Gaussian noise tot he inputs and randomly change the labels'''
    X = X + np.random.normal(loc=0, scale=x_noise, size=X.shape)
    y_shift = np.random.choice(nr_classes, size=y.shape, p=[1/nr_classes]*nr_classes)
    y = np.where(np.random.random(size=y.shape) < y_prob, y_shift, y)
    return X, y