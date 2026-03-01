# Import operating system utilities for file and directory operations
import os
# Import OpenCV library for computer vision and image processing
import cv2
# Import JSON library for handling JSON data serialization
import json
# Import random for generating random numbers
import random
# Import NumPy for numerical computing and array operations
import numpy as np
# Import MediaPipe for machine learning solutions like pose detection
import mediapipe as mp

# Import sys to manipulate Python path
import sys
# Add parent directory to path to import custom modules
sys.path.append("../")
# Import custom LandmarkExtractor class for extracting body landmarks
from landmark_extractor import LandmarkExtractor

'''
MONTREAL Dataset structure
citation: http://www.iro.umontreal.ca/~labimage/Dataset/
24 folders, one for each scenario

Each folder has the same scene recorded from different point of views
'''


def main():
    """Main function to process Montreal dataset and extract body landmarks from videos"""
    # Initialize landmark extractor object
    landmark_extractor = LandmarkExtractor()

    # Store landmarks from previous frame for interpolation when detection fails
    previous_landmarks = None

    # Define sequence length for temporal data chunks
    sequence_length = 90
    # Counter to track processed frames
    frames_counter = 0

    # List to store all processed data sequences
    data = []

    # Path to the Montreal dataset directory
    dataset_path = "~/Downloads/montreal_dataset"
    # Iterate through each scenario folder in the dataset
    for folder in sorted(os.listdir(dataset_path)):
        # Process each experiment (in a separate folder)
        # Queue to store feature vectors for current sequence
        queue = []
        # Body landmarks buffer for current sequence
        bl = []
        # Labels buffer for current sequence
        labels = []
        # Images buffer for current sequence
        images = []

        # Iterate through video sequences in each folder
        for video_sequence in sorted(os.listdir(os.path.join(dataset_path, folder))):
            # Print current processing status
            print(f"Processing sequence {os.path.join(folder, video_sequence)}")
            # Open video file with OpenCV
            video_cap = cv2.VideoCapture(os.path.join(dataset_path, folder, video_sequence))
            # Read first frame from video
            ret, image = video_cap.read()
            
            # Counter to skip frames (videos recorded at 120 Hz, downsample to 1/4)
            skip_counter = 0
            # Index to track frame number in sequence
            frames_indexes = 1
            # Flag to determine if current frame should be processed
            process = True # Videos are recorder at 120 Hz, so downsaple and read 1/4 frames

            # Process all frames in video
            while ret:
                # Check if current frame should be processed
                if process:
                    # Print frame processing status
                    print(f"Processing frame {frames_indexes}")
                    # Extract body landmarks and aspect ratio from current frame
                    body_landmarks, body_ar = landmark_extractor.get_body_landmarks(image)
                    # Check if landmarks were successfully detected
                    if body_landmarks:
                        # Fix wrist landmarks that may not be detected by hand detector
                        body_landmarks = fix_wrist_landmarks(body_landmarks)

                        # Validate that all landmarks are within image bounds
                        check, keys = check_body_landmarks(body_landmarks)
                        # Print warning if landmarks are out of bounds
                        if not check:
                            print(f"{keys} landmarks are out of image!")
                        # Store current landmarks for next frame interpolation
                        previous_landmarks = body_landmarks

                    else:
                        # Use previous frame landmarks if detection fails
                        body_landmarks = previous_landmarks
                    
                    # Append to buffers only if landmarks exist
                    if body_landmarks is not None:
                        # Convert landmarks to feature vector for ML model
                        feature_vector = vectorize_landmarks(body_landmarks, body_ar)
                        # Append feature vector to queue
                        queue.append(feature_vector)
                        # Append body landmarks to buffer
                        bl.append(body_landmarks)
                        # Append label to buffer
                        labels.append(1)
                        # Append image to buffer for visualization
                        images.append(image)

                # Read next frame from video
                ret, image = video_cap.read()
                # Increment frame index counter
                frames_indexes += 1
                # Control frame skipping logic to downsample video
                if skip_counter < 3:
                    # Skip this frame
                    process = False
                    # Increment skip counter
                    skip_counter += 1
                
                # Process next frame after skipping 3 frames
                elif skip_counter == 3:
                    # Process this frame
                    process = True
                    # Reset skip counter
                    skip_counter = 0
                
        
            # At the end of a single experiment, check all extracted sequence and decide if consistent or not
            # Iterate through queue with step of 15 frames to create non-overlapping sequences
            for i in range(0, len(queue) - sequence_length + 1, int(15)):
                # Print sequence extraction status
                print("New sequence...")
                # Print frame range for current sequence
                print(f"Going from {i} to {i+sequence_length}")
                # Initialize row to store single data sample
                row = []

                # Append sequence_length frames to row
                for j in range(sequence_length):
                    # Append feature vector to row
                    row.append(queue[i + j])

                    # Get corresponding image for visualization
                    image = images[i+j]

                    # Get corresponding body landmarks
                    body_landmarks = bl[i + j]
                    # De-normalize landmarks to original image coordinates
                    normalized_bl = landmark_extractor.de_normalize_body_landmarks(body_landmarks)
                    # Draw landmarks on image for visualization
                    out_img = landmark_extractor.draw_selected_landmarks(normalized_bl, image)
                    # Display image with landmarks
                    cv2.imshow('', out_img)
                    # Wait 20ms between frames for visualization
                    cv2.waitKey(20)
                
                
                # Ask user if sequence should be saved
                ques = input("Save this sequence? \n")
                # Save sequence with label 0 (negative class)
                if ques == "0":
                    # Append label 0 to row
                    row.append(0)
                    # Append row to dataset
                    data.append(row)

                    # Print confirmation message
                    print("Saved...")
                    # Print current dataset size
                    print(f"Dataset has now {len(data)} samples")
                
                # Save sequence with label 1 (positive class)
                elif ques == "1":
                    # Append label 1 to row
                    row.append(1)
                    # Append row to dataset
                    data.append(row)

                    # Print confirmation message
                    print("Saved...")
                    # Print current dataset size
                    print(f"Dataset has now {len(data)} samples")
                    
                else:
                    # Print message if sequence was discarded
                    print("Discarded")

    # dump_json(data)       



def dump_json(data):
    """Save processed data to JSON file format"""
    # Define output file path for JSON dataset
    out_path = "~/WatchDog/Perception/Dataset/montreal_dataset.json"

    # Open file in write mode
    with open(out_path, 'w') as json_file:
        # Write JSON array opening bracket
        json_file.write('[\n')
        # Iterate through all rows in dataset
        for row_idx, row in enumerate(data):
            # Write inner array opening bracket
            json_file.write('  [\n')
            # Iterate through all vectors in row
            for vector_idx, vector in enumerate(row):
                # Convert vector to JSON string with indentation
                vector_str = '    ' + json.dumps(vector)
                # Add comma if not last vector in row
                if vector_idx < len(row) - 1:
                    json_file.write(f'{vector_str},\n')
                else:
                    json_file.write(f'{vector_str}\n')
            # Add comma if not last row in dataset
            if row_idx < len(data) - 1:
                json_file.write('  ],\n')
            else:
                json_file.write('  ]\n')
        # Write JSON array closing bracket
        json_file.write(']\n')



def vectorize_landmarks(landmarks, body_ar):
    """Convert landmark dictionary to feature vector for ML model"""
    # Initialize empty input vector
    input_vector = []
    # Iterate through all landmark keys
    for key in landmarks.keys():
        # Append x-coordinate to vector
        input_vector.append(float(landmarks[key][0]))
        # Append y-coordinate to vector
        input_vector.append(float(landmarks[key][1]))

    # Append body aspect ratio to vector
    input_vector.append(body_ar)
    # Return complete feature vector
    return input_vector
    

def fix_wrist_landmarks(landmarks):
    '''
    Hand landmark detector does not capture wrist landmarks, in this case the best thing is to assume person falling with
    straight arms pointing the floor, so wrist landmarks can be similar to hip landmarks
    '''
    # Generate random x-offset for wrist landmarks
    rand_x = random.uniform(0.001, 0.01)
    # Generate random y-offset for wrist landmarks
    rand_y = random.uniform(0.001, 0.01)

    # Check if left wrist was not detected (coordinates are [0, 0])
    if landmarks['left_wrist'] == [0, 0]:
        # Set left wrist x-coordinate to left hip x with random offset
        landmarks['left_wrist'][0] = landmarks['left_hip'][0] + rand_x
        # Set left wrist y-coordinate to left hip y with random offset
        landmarks['left_wrist'][1] = landmarks['left_hip'][1] + rand_y
    # Check if right wrist was not detected
    if landmarks['right_wrist'] == [0, 0]:
        # Set right wrist x-coordinate to right hip x with random offset
        landmarks['right_wrist'][0] = landmarks['right_hip'][0] + rand_x
        # Set right wrist y-coordinate to right hip y with random offset
        landmarks['right_wrist'][1] = landmarks['right_hip'][1] + rand_y
    # Return corrected landmarks dictionary
    return landmarks


def check_body_landmarks(coords):
    """Validate that all body landmarks are within image bounds and valid"""
    # Iterate through all landmark keys in coordinates dictionary
    for key in coords.keys():
        # Iterate through x and y coordinates for each landmark
        for landmark in coords[key]:
            # Check if x-coordinate is None (invalid detection)
            if coords[key][0] is None:
                # Return False (invalid) and the problematic key
                return False, key
            # Check if landmark coordinates are both zero (not detected)
            elif coords[key][0] == float(0) and coords[key][1] == float(0):
                # Return False (invalid) and the problematic key
                return False, key
    # Return True (valid) if all landmarks passed validation
    return True, "None"


# Entry point of the script
if __name__ == "__main__":
    # Call main function to start dataset processing
    main()