"""GRU models.
"""

import os
import torch
from torch import nn
from lln.models.Model import Model

class GRUNet(nn.Module):

    def __init__(self, feature_dim, input_dim, hidden_dim, output_dim, n_layers, seq2seq, drop_prob=0.0):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.seq2seq = seq2seq
        self.batch_first = True
        self.fc1 = nn.Linear(input_dim, feature_dim)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(feature_dim, hidden_dim, n_layers, batch_first=self.batch_first, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h):
        x = self.relu(self.fc1(x))
        out, h = self.gru(x, h)
        out = self.fc(out[:, -1])
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()
        return hidden