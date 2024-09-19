"""A general class for PyTorch models, mainly for saving and restoring.
"""

import os
import torch
from torch import nn

class Model(nn.Module):
    def __init__(self, save_path=None, device='cpu'):
        super().__init__()
        self.name = self.__class__.__name__
        self.save_path = save_path
        self.device = device
        if save_path is not None and not os.path.exists(save_path):
            os.makedirs(save_path)
        self.init()
        
    def save(self, state_name='last', verbose=False):
        '''Saves a model state in the defined path, with the model name'''
        if self.save_path is not None:
            model_state_name = self.name+'_'+state_name+'.pth'
            torch.save(self.state_dict(), os.path.join(self.save_path, model_state_name))
            if verbose:
                print("Saved PyTorch model state {} in {}".format(model_state_name, self.save_path))
            
    def restore(self, state_name):
        '''Restores a model state for the given state name'''
        model_state_name = self.name+'_'+state_name+'.pth'
        self.load_state_dict(torch.load(os.path.join(self.save_path, model_state_name)))
        
    def init(self):
        '''Optional initialization operations'''
        pass
    
    def init_weights(self):
        '''Initialize weights and biases of the model.'''
        for name, param in self.named_parameters():
            # Apply to weights of multi-dimensional parameters (e.g., linear layers, but not biases)
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def forward(self, x):
        '''Forward pass, to be implemented by subclasses'''
        pass
        