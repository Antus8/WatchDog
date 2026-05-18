# Main model definition for body classification using transformer architecture

# Import PyTorch library for neural network operations
import torch
# Import PyTorch neural network module for building layers
import torch.nn as nn
# Import custom Transformer class from local module
from transformer import Transformer


# Define a neural network class for body classification using transformer architecture
class BodyClassificationNetwork(nn.Module):
    # Initialize the network with specified hyperparameters
    def __init__(self, num_transformer_blocks, d_input, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, inner_df):
        # Call parent class constructor
        super(BodyClassificationNetwork, self).__init__()

        # Linear layer to embed input features to model dimension
        self.input_embedding = nn.Linear(d_input, d_model)
        # Create multiple transformer blocks for sequence processing
        self.transformer_blocks = nn.ModuleList([Transformer(d_model, num_heads, num_layers, d_ff, max_seq_length, dropout) for _ in range(num_transformer_blocks)])

        # Adaptive pooling layer to aggregate sequence dimension to single value
        self.pooling = nn.AdaptiveAvgPool1d(1)
        # First fully connected layer for classification head
        self.fc1 = nn.Linear(d_model, inner_df)
        # Output layer producing single classification score
        self.fc2 = nn.Linear(inner_df, 1)
    
    # Forward pass through the network
    def forward(self, x):
        # Embed input features to model dimension
        x = self.input_embedding(x)
        # Initialize transformer output
        trans_out = x
        # Pass through each transformer block sequentially
        for transformer in self.transformer_blocks:
            trans_out = transformer(trans_out)

        # Apply average pooling over sequence length and squeeze output
        out = self.pooling(trans_out.permute(0,2,1)).squeeze(-1)
        # Pass through first classification layer
        out = self.fc1(out)
        # Pass through output layer
        out = self.fc2(out)

        # Apply sigmoid activation for binary classification probability
        return torch.sigmoid(out)