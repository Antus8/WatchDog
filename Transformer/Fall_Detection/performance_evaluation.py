# Import operating system utilities
import os
# Import PyTorch deep learning framework
import torch
# Import NumPy for numerical computations
import numpy as np
# Import Seaborn for statistical data visualization
import seaborn as sns
# Import Matplotlib for plotting
import matplotlib.pyplot as plt
# Import custom BodyClassificationNetwork model
from model import BodyClassificationNetwork
# Import DataLoader for batch processing and random_split for dataset splitting
from torch.utils.data import DataLoader, random_split
# Import custom dataset class for body landmarks
from body_landmarks_dataset import BodyLandmarksDataset
# Import metrics for model evaluation (precision, recall, F1 score, accuracy, confusion matrix)
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix


def main():
    """Main function to evaluate model performance on validation dataset."""
    # Set device to GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dictionary containing model hyperparameters and training configuration
    options = {'num_transformer_blocks':2,
               'd_input':23,
               'd_model':32,
               'num_heads':4,
               'num_layers':2,
               'd_ff':1024,
               'max_seq_length':90,
               'dropout':0.1,
               'inner_df':64,
               'batch_size':32,
               'num_epochs':1000}

    # Path to the normalized dataset JSON file
    dataset_path = '~/WatchDog/Perception/Dataset/final_normalized_dataset.json'
    # Load the body landmarks dataset
    dataset = BodyLandmarksDataset(dataset_path)
    # Split dataset into training and validation sets
    _, val_dataloader = train_val_splitting(dataset, options)

    # Instantiate the model with specified hyperparameters
    model = BodyClassificationNetwork(options['num_transformer_blocks'], 
                                      options['d_input'], 
                                      options['d_model'], 
                                      options['num_heads'], 
                                      options['num_layers'], 
                                      options['d_ff'], 
                                      options['max_seq_length'], 
                                      options['dropout'], 
                                      options['inner_df'])

    # Path to the pre-trained model weights
    weights_path = "../runs/train/weights/best_model.pth"
    # Load pre-trained weights into the model
    model.load_state_dict(torch.load(weights_path))
    # Move model to the specified device (GPU or CPU)
    model.to(device)
    # Set model to evaluation mode (disables dropout, batch norm updates)
    model.eval()

    # Lists to store metrics across multiple evaluation runs
    global_accuracy, global_recall, global_f1 = [], [], []

    # Run 30 evaluation iterations
    for i in range(30):

        # Lists to store all predictions and labels for the current iteration
        all_labels = []
        all_preds = []

        # Disable gradient computation for inference
        with torch.no_grad():
            # Iterate through validation batches
            for idx, features, labels in val_dataloader:
                # Move features and labels to the specified device
                features, labels = features.to(device), labels.to(device)

                # Forward pass through the model
                outputs = model(features)
                # Apply threshold (0.5) to convert outputs to binary predictions
                preds = (outputs > 0.5).float()

                # Append labels and predictions to lists
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        # Convert lists to NumPy arrays for metric computation
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)

        # Calculate accuracy on validation set
        accuracy = accuracy_score(all_labels, all_preds)
        # Calculate precision for binary classification
        precision = precision_score(all_labels, all_preds, average='binary')
        # Calculate recall for binary classification
        recall = recall_score(all_labels, all_preds, average='binary')
        # Calculate F1 score for binary classification
        f1 = f1_score(all_labels, all_preds, average='binary')

        # print(f"Validation Accuracy: {accuracy}")
        # print(f"Validation Precision: {precision}")
        # print(f"Validation Recall: {recall}")
        # print(f"Validation F1 Score: {f1}")

        # Compute the confusion matrix
        # conf_matrix = confusion_matrix(all_labels, all_preds)

        # plt.figure(figsize=(6, 6))
        # sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
        #             xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
        # plt.xlabel('Predicted')
        # plt.ylabel('True')
        # plt.title('Confusion Matrix')
        # plt.show()

        # Append current iteration's metrics to global lists
        global_accuracy.append(accuracy)
        global_recall.append(recall)
        global_f1.append(f1)

    # Print average accuracy across all 30 iterations
    print(f"AVG Validation Accuracy: {np.average(accuracy)}")
    # Print average precision across all 30 iterations
    print(f"AVG Validation Precision: {np.average(precision)}")
    # Print average recall across all 30 iterations
    print(f"AVG Validation Recall: {np.average(recall)}")
    # Print average F1 score across all 30 iterations
    print(f"AVG Validation F1 Score: {np.average(f1)}")
    

    

def train_val_splitting(dataset, options):
    """
    Split dataset into training and validation subsets.
    
    Args:
        dataset: The full dataset to split
        options: Dictionary containing hyperparameters (includes 'batch_size')
    
    Returns:
        tuple: (train_loader, val_loader) - DataLoaders for training and validation
    """
    # Total size of the dataset
    dataset_size = 2618
    # Fraction of data to use for validation
    val_split = 0.2  
    # Calculate number of samples for validation set
    val_size = int(val_split * dataset_size)
    # Calculate number of samples for training set
    train_size = dataset_size - val_size

    # Split dataset into training and validation subsets using random_split
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoader for training set with specified batch size
    train_loader = DataLoader(train_dataset, batch_size=options['batch_size'], shuffle=False)
    # Create DataLoader for validation set with specified batch size
    val_loader = DataLoader(val_dataset, batch_size=options['batch_size'], shuffle=False)

    # Return both dataloaders
    return train_loader, val_loader


# Entry point of the script
if __name__ == "__main__":
    # Call main function to start evaluation
    main()