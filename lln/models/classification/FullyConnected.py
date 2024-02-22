"""Classifiers made excludively out of fully-connected layers.
"""

from torch import nn
from lln.models.classification.Classifier import Classifier
import torch.nn.functional as F

class FullyConnected(Classifier):
    def __init__(self, *args, input_size=28, **kwargs):
        super(FullyConnected, self).__init__(*args, **kwargs)
        self.flatten = nn.Flatten()
        self.input_size = input_size
        self.linear_layers = lambda *args: None # To overwrite
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_layers(x)
        return logits

class FullyConnected3(FullyConnected):
    def __init__(self, *args, dropout=0, **kwargs):
        super(FullyConnected3, self).__init__(*args, **kwargs)
        self.linear_layers = nn.Sequential(
            nn.Linear(self.input_size, 10),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(5, len(self.labels)),
        )
    
class FullyConnected5(FullyConnected):
    def __init__(self, *args, dropout=0, **kwargs):
        super(FullyConnected5, self).__init__(*args, **kwargs)
        self.linear_layers = nn.Sequential(
            nn.Linear(self.input_size, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, len(self.labels)),
        )
    