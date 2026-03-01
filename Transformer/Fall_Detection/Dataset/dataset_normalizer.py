# Import the json module for reading and writing JSON files
import json
# Import numpy library and alias it as np (note: numpy is imported but not used in this code)
import numpy as np


# Class definition for normalizing dataset values
class DatasetNormalizer:
    # Constructor method to initialize file paths
    def __init__(self):
        # Path to the input JSON file containing the original dataset
        self.in_path = "~/WatchDog/Perception/Dataset/final_merged_dataset.json"
        # Path to the output JSON file where normalized dataset will be saved
        self.out_path = "~/WatchDog/Perception/Dataset/final_normalized_dataset.json"

    # Method to normalize values in the dataset using min-max normalization
    def normalize(self):
        # Open and read the input JSON file
        with open(self.in_path, 'r') as json_file:
            data = json.load(json_file)

        # Extract minimum and maximum values from the dataset (specifically at index 22)
        min_bar, max_bar = self.extract_min_max(data)
        
        # Iterate through each row in the dataset
        for row in data:
            # Iterate through vectors in each row, excluding the last element (label)
            for vector in row[:-1]:
                # Get the value at index 22 from the vector
                value = vector[22]
                # Apply min-max normalization formula: (value - min) / (max - min)
                normalized_value = (value - min_bar) / (max_bar - min_bar)
                # Replace the original value with the normalized value
                vector[22] = normalized_value

        # Save the normalized dataset to the output file
        self.dump_dataset(data)

    # Method to verify that the normalized dataset contains only values between 0 and 1
    def check_normalized_dataset(self):
        # Open and read the normalized output JSON file
        with open(self.out_path, 'r') as json_file:
            data = json.load(json_file)

        # Print the number of rows in the dataset
        print(len(data))
        # Print the number of vectors in the first row
        print(len(data[0]))
        # Print the number of elements in the first vector of the first row
        print(len(data[0][0]))

        # Iterate through each row in the dataset
        for row in data:
            # Iterate through vectors in each row, excluding the last element
            for vector in row[:-1]:
                # Iterate through each element in the vector
                for el in vector:
                    # Check if element is outside the normalized range [0, 1]
                    if (el < 0) or (el > 1):
                        # Print the out-of-range element
                        print(el)
                        # Return False indicating normalization failed
                        return False

        # Return True if all values are within [0, 1] range
        return True

    # Method to write the dataset to a formatted JSON file
    def dump_dataset(self, data):
        # Open the output file for writing
        with open(self.out_path, 'w') as json_file:
            # Write opening bracket for JSON array
            json_file.write('[\n')
            # Iterate through each row with its index
            for row_idx, row in enumerate(data):
                # Write opening bracket for row array
                json_file.write('  [\n')
                # Iterate through each vector in the row with its index
                for vector_idx, vector in enumerate(row):
                    # Convert vector to JSON string with indentation
                    vector_str = '    ' + json.dumps(vector)
                    # If not the last vector in the row, add a comma
                    if vector_idx < len(row) - 1:
                        json_file.write(f'{vector_str},\n')
                    # For the last vector, don't add a comma
                    else:
                        json_file.write(f'{vector_str}\n')
                # If not the last row, add a comma after closing bracket
                if row_idx < len(data) - 1:
                    json_file.write('  ],\n')
                # For the last row, don't add a comma
                else:
                    json_file.write('  ]\n')
            # Write closing bracket for JSON array
            json_file.write(']\n')

    # Method to find the minimum and maximum values at index 22 across all vectors
    def extract_min_max(self, data):
        # Initialize minimum value to positive infinity
        min_value = float('inf')
        # Initialize maximum value to negative infinity
        max_value = float('-inf')

        # Iterate through each row in the dataset
        for row in data:
            # Iterate through vectors in each row, excluding the last element (label)
            for vector in row[:-1]:
                # Get the value at index 22 (the 23rd element)
                bar = vector[22]
                # Update minimum value if current value is smaller
                if bar < min_value:
                    min_value = bar
                # Update maximum value if current value is larger
                if bar > max_value:
                    max_value = bar
        
        # Return the minimum and maximum values found
        return min_value, max_value


# Entry point: execute when script is run directly
if __name__ == "__main__":
    # Create an instance of the DatasetNormalizer class
    dataset_normalizer = DatasetNormalizer()
    # Uncomment the line below to run the normalize method
    # dataset_normalizer.normalize()
    # Call check_normalized_dataset and print the result (True if valid, False otherwise)
    print(dataset_normalizer.check_normalized_dataset())