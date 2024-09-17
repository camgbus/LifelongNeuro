"""LSTM models.
"""

import torch
from torch import nn
from lln.models.Model import Model
import numpy as np

class LSTM(Model):
    
    def __init__(self, input_dim, hidden_dim, output_dim, nr_layers, *args, regression=False, seq_to_seq=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim
        self.nr_layers = nr_layers
        self.seq_to_seq = seq_to_seq
        self.hidden_states = {}
        self.regression = regression
        
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
            if self.regression:
                return pred.detach().cpu().numpy()
            else:
                return int(pred.argmax().detach())
    
    def register_hooks(self):
        self.hidden_states = {}
        def lstm_hook(module, input, output):
            # output is a tuple (out, (hn, cn))
            lstm_output, (hn, cn) = output
            self.hidden_states["lstm_output"] = lstm_output.detach()
            self.hidden_states["hidden_state"] = hn.detach()
            self.hidden_states["cell_state"] = cn.detach()
        self.lstm.register_forward_hook(lstm_hook)
        
    def get_hooked_hidden_states(self):
        for key, value in self.hidden_states.items():
            self.hidden_states[key] = np.array(value)
        return self.hidden_states