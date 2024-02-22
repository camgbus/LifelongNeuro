"""LSTM models.
"""

import torch
from torch import nn
from lln.models.Model import Model

class LSTM(Model):
    
    def __init__(self, input_dim, hidden_dim, output_dim, nr_layers, *args, seq_to_seq=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim
        self.nr_layers = nr_layers
        self.seq_to_seq = seq_to_seq
        
        # LSTM layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, nr_layers, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.nr_layers, x.size(0), self.hidden_dim).requires_grad_()
        
        # Initialize cell state
        c0 = torch.zeros(self.nr_layers, x.size(0), self.hidden_dim).requires_grad_()
        
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        
        if self.seq_to_seq:
            out = self.fc(out)
            out = out.permute(0, 2, 1)  # [batch_size, seq_length, num_classes] -> [batch_size, num_classes, seq_length]
        else:
            # Return only last time step
            out = self.fc(out[:, -1, :]) 
        
        return out
    
    def predict(self, X):
        '''Make a prediction based on a given input'''
        self.eval()
        with torch.no_grad():
            pred = self(X)
            return int(pred.argmax().detach())