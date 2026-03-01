import math
import torch
import torch.nn as nn


# Define a positional encoding module for transformer-based models
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        # Initialize the parent nn.Module class
        super(PositionalEncoding, self).__init__()
        
        # Create a tensor of zeros with shape (max_seq_length, d_model)
        pe = torch.zeros(max_seq_length, d_model)
        # Create position indices from 0 to max_seq_length, reshape to (max_seq_length, 1)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        # Calculate the division term for the positional encoding formula
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        '''
        Each position to have a unique combination of sine and cosine values, encoding the position information in a continuous space
        '''
        # Apply sine to even indices (0, 2, 4, ...)
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices (1, 3, 5, ...)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register pe as a buffer so it moves to GPU/CPU with the model but isn't a trainable parameter
        self.register_buffer('pe', pe.unsqueeze(0)) # these parameters won't be updated during training
        

    def forward(self, x):
        # Add positional encoding to input tensor and return the result
        return x + self.pe[:, :x.size(1)]