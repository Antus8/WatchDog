# Import the neural network module from PyTorch
import torch.nn as nn


# Define a position-wise feed-forward network module
class PositionWiseFeedForward(nn.Module):
    # Initialize the feed-forward network with model dimension and feed-forward dimension
    def __init__(self, d_model, d_ff):
        # Call parent class constructor
        super(PositionWiseFeedForward, self).__init__()
        # First linear layer: project from d_model to d_ff (expansion)
        self.fc1 = nn.Linear(d_model, d_ff)
        # Second linear layer: project from d_ff back to d_model (compression)
        self.fc2 = nn.Linear(d_ff, d_model)
        # ReLU activation function for non-linearity
        self.relu = nn.ReLU()

    # Forward pass through the network
    def forward(self, x):
        # Apply first linear transformation
        x = self.fc1(x)
        # Apply ReLU activation
        x = self.relu(x)
        # Apply second linear transformation and return output
        out = self.fc2(x)
        return out