# Import the os module for file and directory operations
import os
# Import the json module for reading and writing JSON files
import json
# Import the random module for shuffling data
import random

# Define the main function that orchestrates the dataset merging process
def main():
    # Specify the input directory containing JSON dataset files
    input_directory = "~/WatchDog/Perception/Dataset/Data_Files"
    # Specify the output path for the merged dataset
    out_path = "~/WatchDog/Perception/Dataset/final_merged_dataset.json"
    # Initialize an empty list to store all dataset samples
    global_dataset = []
    # Initialize counter for total number of samples
    tot_samples = 0

    # Iterate through all files in the input directory
    for file in os.listdir(input_directory):
        # Construct the full file path
        file_path = os.path.join(input_directory, file)

        # Open and read the JSON file
        with open(file_path, 'r') as json_file:
            # Load the JSON data into a Python object
            data = json.load(json_file)
        # Accumulate the count of samples
        tot_samples = tot_samples + len(data)
        # Initialize variable to store previous body aspect ratio
        previous_bar = None

        # Iterate through each sample in the current dataset
        for sample in data:
            # Iterate through all frames in the sample except the last one
            for frame_data in sample[:-1]:
                # Adjusting Body aspect ratio 

                # If body aspect ratio (index 22) is None, use the previous value
                if frame_data[22] is None:
                    frame_data[22] = previous_bar
                # If body aspect ratio is greater than 40, normalize it by dividing by 10
                elif frame_data[22] > 40:
                    frame_data[22] = frame_data[22]/10 # suppose a 10 px body width 
                
                # Store the current body aspect ratio for next iteration
                previous_bar = frame_data[22]

                # Commented-out validation check for debugging purposes
                # if (frame_data[22] < 0) or  (frame_data[22] > 40) or (frame_data[22] is None):
                #     print("Error")
        
            # Add the processed sample to the global dataset
            global_dataset.append(sample)
    
    # Verify that all samples have been collected
    if len(global_dataset) == tot_samples:
        # Shuffle the dataset once
        random.shuffle(global_dataset)
        # Shuffle the dataset a second time for better randomization
        random.shuffle(global_dataset)
        
        # Check dimensions after shuffling
        # Print the total number of samples
        print(len(global_dataset))
        # Print the number of frames in the first sample
        print(len(global_dataset[0]))
        # Print the number of data points in the first frame of the first sample
        print(len(global_dataset[0][0]))

        # Write the merged dataset to the output file
        dump_merged_dataset(global_dataset, out_path)
        # Print confirmation message
        print("Ok...")


# Define function to write the dataset to a JSON file with custom formatting
def dump_merged_dataset(data, out_path):
    # Open the output file in write mode
    with open(out_path, 'w') as json_file:
        # Write opening bracket for JSON array
        json_file.write('[\n')
        # Iterate through each sample with its index
        for row_idx, row in enumerate(data):
            # Write opening bracket for sample array
            json_file.write('  [\n')
            # Iterate through each frame/vector in the sample with its index
            for vector_idx, vector in enumerate(row):
                # Convert the vector to a JSON string with proper indentation
                vector_str = '    ' + json.dumps(vector)
                # If not the last vector in the sample, add a comma
                if vector_idx < len(row) - 1:
                    json_file.write(f'{vector_str},\n')
                # If last vector in the sample, do not add a comma
                else:
                    json_file.write(f'{vector_str}\n')
            # If not the last sample, add a comma after closing bracket
            if row_idx < len(data) - 1:
                json_file.write('  ],\n')
            # If last sample, do not add a comma after closing bracket
            else:
                json_file.write('  ]\n')
        # Write closing bracket for JSON array
        json_file.write(']\n')



# Standard Python idiom to execute main function when script is run directly
if __name__ == "__main__":
    main()
