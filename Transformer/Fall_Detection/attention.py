# Import math module for mathematical operations
import math
# Import PyTorch library for tensor operations
import torch
# Import PyTorch neural network module for building layers
import torch.nn as nn


# Define MultiHeadAttention class inheriting from nn.Module
class MultiHeadAttention(nn.Module):
    # Initialize the module with model dimension and number of attention heads
    def __init__(self, d_model, num_heads):
        # Call parent class constructor
        super(MultiHeadAttention, self).__init__()
        
        # d_model: size of each input vector (e.g., 64 after dimensionality reduction)
        # num_heads: number of parallel attention heads
        # Input shape: [batch_size, sequence_length=90, feature_dim=23]
        
        # Store the model dimension
        self.d_model = d_model
        # Store the number of attention heads
        self.num_heads = num_heads
        # Calculate dimension of each attention head
        self.d_k = d_model // num_heads
        
        # Linear transformation for Query projection
        self.W_q = nn.Linear(d_model, d_model)
        # Linear transformation for Key projection
        self.W_k = nn.Linear(d_model, d_model)
        # Linear transformation for Value projection
        self.W_v = nn.Linear(d_model, d_model)
        # Linear transformation for output projection
        self.W_o = nn.Linear(d_model, d_model)

    # Compute scaled dot-product attention
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Compute attention scores: (Q @ K^T) / sqrt(d_k) for numerical stability
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # Apply mask to prevent attention to certain positions (optional)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        # Convert scores to probabilities using softmax
        attn_probs = torch.softmax(attn_scores, dim=-1)
        # Apply attention weights to Values
        output = torch.matmul(attn_probs, V)
        # Return the attention output
        return output
    
    # Split input into multiple heads for parallel processing
    def split_heads(self, x):
        # Reshape input to separate attention heads
        # Input: [batch_size, seq_length, d_model]
        batch_size, seq_length, d_model = x.size()
        # Reshape to [batch_size, seq_length, num_heads, d_k] then transpose to [batch_size, num_heads, seq_length, d_k]
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    # Recombine multiple heads into single tensor
    def combine_heads(self, x):
        # Get dimensions: [batch_size, num_heads, seq_length, d_k]
        batch_size, _, seq_length, d_k = x.size()
        # Transpose back to [batch_size, seq_length, num_heads, d_k] then reshape to [batch_size, seq_length, d_model]
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    # Forward pass through multi-head attention
    def forward(self, Q, K, V, mask=None):
        # Project and split Query into multiple heads
        Q = self.split_heads(self.W_q(Q))
        # Project and split Key into multiple heads
        K = self.split_heads(self.W_k(K))
        # Project and split Value into multiple heads
        V = self.split_heads(self.W_v(V))
        
        # Compute scaled dot-product attention across all heads
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        # Recombine heads and apply final output projection
        output = self.W_o(self.combine_heads(attn_output))
        # Return the final attention output
        return output