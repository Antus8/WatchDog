import json
# Import json module for reading JSON files
import torch
# Import PyTorch library for tensor operations
from torch.utils.data import Dataset
# Import Dataset base class from PyTorch for creating custom datasets

class BodyLandmarksDataset(Dataset):
    """
    Custom PyTorch Dataset class for loading body landmark data from JSON files.
    Extracts landmark coordinates and prepares them for model training.
    """

    def __init__(self, json_file_path):
        # Constructor: initialize dataset with path to JSON file
        super(BodyLandmarksDataset, self).__init__()
        # Call parent Dataset class constructor

        with open(json_file_path, 'r') as json_file:
            # Open and read the JSON file in read mode
            self.data = json.load(json_file)
            # Load JSON data as list of samples

        self.dataset = []
        # Initialize empty list to store processed samples
        temp_sample = []
        # Initialize temporary list for current sample processing

        for sample in self.data:
            # Iterate through each sample in the loaded data
            for i in range(0, 91, 3):
                # Extract every 3rd element (step=3) from indices 0-90
                temp_sample.append(sample[i])
                # Append extracted landmark coordinate to temporary sample

            self.dataset.append(temp_sample)
            # Add processed sample to main dataset list
            temp_sample = []
            # Reset temporary sample for next iteration

        print(f"LOADED DATASET AT 10Hz WITH {len(self.dataset)} SAMPLES, HAVING {len(self.dataset[0])} TS EACH...")
        # Print dataset statistics
            

    def __len__(self):
        # Return total number of samples in dataset
        return len(self.data)

    def __getitem__(self, idx):
        '''
        Extract data in torch.tensor format 
        features are tensors of shape [sequence_length, num_features_per_vector]
        '''
        
        frame_data = self.dataset[idx]
        # Retrieve the sample at index idx
        
        features = torch.tensor([vector for vector in frame_data[:-1]], dtype=torch.float32)
        # Convert all vectors except the last one to PyTorch tensor with float32 dtype
        label = torch.tensor(frame_data[-1], dtype=torch.float32)
        # Convert last vector element as label (target) to PyTorch tensor with float32 dtype
        
        return idx, features, label
        # Return sample index, features, and label as tuple
