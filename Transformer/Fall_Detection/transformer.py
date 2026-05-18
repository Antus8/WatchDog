# Import math module for mathematical operations
import math
# Import PyTorch library for deep learning
import torch
# Import PyTorch neural network module
import torch.nn as nn
# Import custom EncoderLayer class from local module
from encoder import EncoderLayer
# Import custom PositionalEncoding class from local module
from positional_encoding import PositionalEncoding


# Define Transformer class that inherits from nn.Module
class Transformer(nn.Module):
    # Initialize Transformer with model parameters
    def __init__(self, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        # Call parent class constructor
        super(Transformer, self).__init__()
        
        # Create positional encoding layer with specified dimensions and sequence length
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        # Create a list of encoder layers with specified number of layers
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        

    # Forward pass through the Transformer
    def forward(self, x):
        # Apply positional encoding to input tensor
        x = self.positional_encoding(x)

        # Initialize encoder output with positionally encoded input
        enc_output = x
        # Pass input through each encoder layer sequentially
        for enc_layer in self.encoder_layers:
            # Apply current encoder layer (mask is None for this implementation)
            enc_output = enc_layer(enc_output, None)

        # Return final encoder output
        return enc_output