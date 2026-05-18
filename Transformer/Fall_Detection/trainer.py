# Import operating system utilities
import os
# Import PyTorch
import torch
# Import PyTorch neural network modules
import torch.nn as nn
# Import PyTorch optimization algorithms
import torch.optim as optim
# Import the custom body classification model
from model import BodyClassificationNetwork
# Import data loading utilities and dataset splitting function
from torch.utils.data import DataLoader, random_split
# Import TensorBoard writer for logging training metrics
from torch.utils.tensorboard import SummaryWriter 
# Import the custom dataset class for body landmarks
from body_landmarks_dataset import BodyLandmarksDataset


# Main training function
def train():
    # Initialize experiment index for tracking multiple runs
    experiment_index = 1
    # Set device to GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Define hyperparameters and model configuration options
    options = {'num_transformer_blocks':2,
               'd_input':23,
               'd_model':32,
               'num_heads':4,
               'num_layers':2,
               'd_ff':1024,
               'max_seq_length':30, # 90,
               'dropout':0.1,
               'inner_df':64,
               'batch_size':32,
               'num_epochs':1000}

    # Instantiate the body classification model with specified options
    model = BodyClassificationNetwork(options['num_transformer_blocks'], 
                                      options['d_input'], 
                                      options['d_model'], 
                                      options['num_heads'], 
                                      options['num_layers'], 
                                      options['d_ff'], 
                                      options['max_seq_length'], 
                                      options['dropout'], 
                                      options['inner_df'])
    # Move model to the specified device (GPU or CPU)
    model.to(device)

    # Define binary cross-entropy loss function for training
    criterion = nn.BCELoss()
    # Initialize Adam optimizer with learning rate and epsilon parameters
    optimizer = optim.Adam(model.parameters(), lr=0.0001, eps=1e-9)

    # Specify path to the dataset JSON file
    dataset_path = '../Dataset/final_normalized_dataset.json'
    # Load the body landmarks dataset
    dataset = BodyLandmarksDataset(dataset_path)

    # Split dataset into training and validation sets
    # dataloader = DataLoader(dataset, batch_size=options['batch_size'], shuffle=False)
    train_dataloader, val_dataloader = train_val_splitting(dataset, options)
    
    # Set model to training mode
    model.train()

    # Initialize TensorBoard writer for logging training progress
    writer = SummaryWriter('runs/train_' + str(experiment_index)) # http://localhost:6006
    # Define path for saving model weights
    weights_path = './runs/train_' + str(experiment_index) + "/weights"
    # Create weights directory if it doesn't exist
    if not os.path.exists(weights_path):
        os.mkdir(weights_path)

    # Save model checkpoint every N epochs
    save_epoch_step = 20
    # Initialize best validation loss tracker
    best_val_loss = float('inf')

    # Main training loop over all epochs
    for epoch in range(options['num_epochs']):
        # Initialize epoch loss accumulator
        epoch_loss = 0
        # Iterate through training batches
        for idx, features, labels in train_dataloader:
            # Move features and labels to the specified device
            features, labels = features.to(device), labels.to(device)

            # Forward pass: compute model predictions
            output = model(features)
            
            # Compute loss between predictions and ground truth labels
            loss = criterion(output.squeeze(-1), labels)

            # Backpropagation
            # Clear previous gradients
            optimizer.zero_grad()
            # Compute gradients
            loss.backward()
            # Update model parameters
            optimizer.step()

            # Accumulate loss for the epoch
            epoch_loss += loss.item()

            # Calculate global step for TensorBoard logging
            step = int(epoch * len(train_dataloader) + idx[-1])
            # Log training loss to TensorBoard
            writer.add_scalar('Loss/train', loss.item(), step)
            # Log training accuracy to TensorBoard
            writer.add_scalar('Accuracy/train', ((output > 0.5).float() == labels).float().mean().item(), step)

        # Print training progress
        print(f"Epoch [{epoch+1}/{options['num_epochs']}], Loss: {loss.item():.4f}")
        
        # Set model to evaluation mode
        model.eval()  # Set model to evaluation mode
        # Initialize validation loss accumulator
        val_loss = 0
        # Disable gradient computation for validation
        with torch.no_grad():  # Disable gradient computation
            # Iterate through validation batches
            for idx, features, labels in val_dataloader:
                # Move features and labels to the specified device
                features, labels = features.to(device), labels.to(device)

                # Forward pass: compute model predictions
                output = model(features)
                # Compute validation loss
                loss = criterion(output.squeeze(-1), labels)

                # Accumulate validation loss
                val_loss += loss.item()

                # Calculate global step for TensorBoard logging
                step = int(epoch * len(train_dataloader) + idx[-1])
                # Log validation loss to TensorBoard
                writer.add_scalar('Loss/val', loss.item(), step)
                # Log validation accuracy to TensorBoard
                writer.add_scalar('Accuracy/val', ((output > 0.5).float() == labels).float().mean().item(), step)

        # Calculate average validation loss
        val_loss /= len(val_dataloader)
        # print(f"Epoch [{epoch+1}/{options['num_epochs']}], Validation Loss: {val_loss:.4f}")

        # If current validation loss is better than best, save model
        if val_loss < best_val_loss:
            # Update best validation loss
            best_val_loss = val_loss
            # Define path for saving best model weights
            best_weights_path = os.path.join(weights_path, "best_model.pth")
            # Save best model state dictionary
            torch.save(model.state_dict(), best_weights_path)
            # Print confirmation message
            print(f"Best model saved with validation loss: {best_val_loss:.4f}")

        # Save model checkpoint at regular intervals
        if epoch % save_epoch_step == 0:
            # Define checkpoint file path
            out_path = os.path.join(weights_path, f"model_epoch_{epoch}.pth")
            # Save model state dictionary
            torch.save(model.state_dict(), out_path)

    # Print completion message
    print('Training complete!!!!')


# Function to split dataset into training and validation sets
def train_val_splitting(dataset, options):
    # Define total dataset size
    dataset_size = 2618
    # Set validation split ratio
    val_split = 0.2  
    # Calculate number of validation samples
    val_size = int(val_split * dataset_size)
    # Calculate number of training samples
    train_size = dataset_size - val_size

    # Split dataset into training and validation subsets
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Print training set size
    print(f"TRAIN SPLIT HAS {len(train_dataset)} samples")
    # Print validation set size
    print(f"VAL SPLIT HAS {len(val_dataset)} samples")

    # Create training data loader
    train_loader = DataLoader(train_dataset, batch_size=options['batch_size'], shuffle=False)
    # Create validation data loader
    val_loader = DataLoader(val_dataset, batch_size=options['batch_size'], shuffle=False)

    # Return both data loaders
    return train_loader, val_loader


# Entry point for script execution
if __name__ == "__main__":
    # Call the main training function
    train()