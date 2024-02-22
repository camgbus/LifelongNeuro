"""Manually define a torch.utils.data.Dataset from an events dataframe.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import sys
from torchvision.transforms import ToTensor, Lambda
from collections import defaultdict

class PandasDataset(Dataset):
    def __init__(self, df, feature_cols, target_col, y_dtype=torch.int64):
        self.X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        self.y = torch.tensor(df[target_col].values, dtype=y_dtype)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class LongDataset(Dataset):
    '''Dataset for longitudinal subject-visit data'''	
    def __init__(self, df, feature_cols, target_col, seq_to_seq=False, timepoints=None, 
                 id_col='subject', seq_col='visit', y_dtype=torch.int64):
        self.y_dtype = y_dtype
        
        # Sort data by subject and visit
        data = df.sort_values([id_col, seq_col])
        
        # Organize data by subject
        self.X = defaultdict(list)
        self.y = defaultdict(list)
        
        
        
        # If the timepoints are specified and seq_to_seq=False, the last time point must be present
        subjects_no_target = []
        
        # This tells me what time points need to be present in the dataset, for eventual padding
        if timepoints is not None:
            for id_x in data[id_col].unique():
                for t in timepoints:
                    id_time_data = data[(data[id_col] == id_x) & (data[seq_col] == t)]
                    assert len(id_time_data) <= 1, f"Multiple entries for subject {id_x} at time {t}"
                    # If needed, pad with 0s for certain time points
                    if len(id_time_data) == 0:
                        self.X[id_x].append(np.zeros(len(feature_cols), dtype=np.float32))
                        if seq_to_seq:
                            self.y[id_x].append(0)
                        else:
                            if t == timepoints[-1]:
                                subjects_no_target.append(id_x)
                            self.y[id_x] = 0
                    else:
                        row = id_time_data.iloc[0]
                        self.X[row[id_col]].append(np.array(row[feature_cols].values, dtype=np.float32))
                        if seq_to_seq:
                            self.y[row[id_col]].append(row[target_col])
                        else:
                            self.y[row[id_col]] = row[target_col]
        # If no timepoints are given, simply append those that are there
        else:
            for _, row in data.iterrows():
                self.X[row[id_col]].append(np.array(row[feature_cols].values, dtype=np.float32))
                if seq_to_seq:
                    self.y[row[id_col]].append(row[target_col])
                else:
                    self.y[row[id_col]] = row[target_col]
        
        for subject in list(self.X.keys()):
            if subject in subjects_no_target:
                print(f"Warning: subject {subject} does not have a target at the last time point, so it was removed")
                del self.X[subject]
                del self.y[subject]
            else:
                self.X[subject] = torch.tensor(np.array(self.X[subject]), dtype=torch.float32)
                self.y[subject] = torch.tensor(np.array(self.y[subject]), dtype=y_dtype)
                
        self.subjects = list(self.X.keys())

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subject_id = self.subjects[idx]
        return self.X[subject_id], self.y[subject_id]
    