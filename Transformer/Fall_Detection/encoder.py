# Import PyTorch neural network modules
import torch.nn as nn
# Import custom MultiHeadAttention module
from attention import MultiHeadAttention
# Import custom PositionWiseFeedForward module
from position_ff import PositionWiseFeedForward


# Define an encoder layer class that inherits from nn.Module
class EncoderLayer(nn.Module):
    # Initialize the encoder layer with model dimensions, attention heads, feed-forward size, and dropout rate
    def __init__(self, d_model, num_heads, d_ff, dropout):
        # Call parent class constructor
        super(EncoderLayer, self).__init__()
        # Create multi-head self-attention layer
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        # Create position-wise feed-forward network
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        # Create layer normalization for attention output
        self.norm1 = nn.LayerNorm(d_model)
        # Create layer normalization for feed-forward output
        self.norm2 = nn.LayerNorm(d_model)
        # Create dropout layer to prevent overfitting
        self.dropout = nn.Dropout(dropout)
        
    # Define forward pass through the encoder layer
    def forward(self, x, mask):
        # Apply multi-head self-attention (query, key, value from same input x)
        attn_output = self.self_attn(x, x, x, mask)
        # Apply dropout, add residual connection, and normalize
        x = self.norm1(x + self.dropout(attn_output))
        # Apply position-wise feed-forward network
        ff_output = self.feed_forward(x)
        # Apply dropout, add residual connection, and normalize
        x = self.norm2(x + self.dropout(ff_output))
        # Return processed output
        return x