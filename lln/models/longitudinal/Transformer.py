"""LSTM models.
"""

import torch
from torch import nn
import math
from lln.models.Model import Model

class PositionalEncoding(nn.Module):
    '''Positional encoding added to the input embeddings.'''
    def __init__(self, dim_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_model, 2) * -(math.log(10000.0) / dim_model))
        pe = torch.zeros(max_len, 1, dim_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class FeatureTransformer(Model):
    
    #def __init__(self, input_dim, hidden_dim, output_dim, nr_layers, *args, seq_to_seq=False, **kwargs):
    def __init__(self, input_dim, dim_model, num_heads, dim_feedforward, nr_layers, *args, output_dim=None, dropout=0.1, seq_to_seq=False, see_future=False, **kwargs):
        '''
        A transformer model for longitudinal predictions.
        
        Args:
            input_dim: Input size, e.g. number of features
            dim_model: Dimension of the model
            num_heads: Number of attention heads
            dim_feedforward: Dimension of the feedforward network model
            nr_layers: Number of stacked transformer layers
            output_dim: Nr. of classes for classification
            dropout: Dropout rate
        '''
        super().__init__(*args, **kwargs)
        self.seq_to_seq = seq_to_seq
        self.see_future = see_future
        self.input_projection = nn.Linear(input_dim, dim_model)
        self.pos_encoder = PositionalEncoding(dim_model, dropout)
        
        # Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=dim_model, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layers, num_layers=nr_layers)

        if self.seq_to_seq:
            # Initialize decoder components for sequence-to-sequence predictions
            self.decoder_input_projection = nn.Linear(input_dim, dim_model)
            decoder_layers = nn.TransformerDecoderLayer(d_model=dim_model, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout)
            self.transformer_decoder = nn.TransformerDecoder(decoder_layer=decoder_layers, num_layers=nr_layers)
            self.output_projection = nn.Linear(dim_model, input_dim)  # Assuming output feature size is same as input
        else:
            # Initialize a final linear layer for non-sequence-to-sequence mode to predict the final value
            self.final_layer = nn.Linear(dim_model, output_dim) if output_dim is not None else None
            
        self.dim_model = dim_model
        self.init_weights()
            
    def generate_square_subsequent_mask(self, sz):
        '''Hide all future tokens from being attended to. The mask has zeros in positions where 
        attention is allowed and -inf where it is not (the softmax operation in the attention 
        mechanism will turn these into zeros).'''
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask
        
    def forward(self, src, tgt=None, src_mask=None, tgt_mask=None, memory_mask=None, 
                src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # If no specific mask is provided but we must ignore the future, generate a causal mask
        if src_mask is None and self.seq_to_seq and not self.see_future:
            src_mask = self.generate_square_subsequent_mask(src.size(0)).to(src.device)
        if tgt_mask is None and self.seq_to_seq and not self.see_future:
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(0)).to(tgt.device)

        src = self.input_projection(src)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        if self.seq_to_seq:
            tgt = self.decoder_input_projection(tgt)
            tgt = self.pos_encoder(tgt)
            output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, 
                tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
            output = self.output_projection(output)
        else:
            # For single-output tasks, no need for target sequences or causal masking
            output = memory[:, -1, :]  # Assuming we're interested in the last output for prediction
            if self.final_layer is not None:
                output = self.final_layer(output)

        return output
    
    def predict(self, X):
        '''Make a prediction based on a given input'''
        self.eval()
        with torch.no_grad():
            pred = self(X)
            return int(pred.argmax().detach())