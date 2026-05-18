# Import operating system module for file operations
import os
# Import OpenCV library for image processing
import cv2
# Import JSON module for data serialization
import json
# Import random module for generating random numbers
import random
# Import NumPy library for numerical operations
import numpy as np
# Import MediaPipe library for pose detection
import mediapipe as mp

# Add parent directory to Python path for custom imports
import sys
sys.path.append("../")
# Import custom landmark extractor class
from landmark_extractor import LandmarkExtractor

'''
UR-FALL Dataset structure
citation: http://fenix.ur.edu.pl/~mkepski/ds/uf.html
30 folders, each one  with a falling scenario
Label all activities as '1'

Read from each sigle folder, get landmarks and wrap into temporal sequences of 3 seconds
'''


# Function to process ADL (Activities of Daily Living) videos from the dataset
def process_adl_videos():
    # Initialize landmark extractor
    landmark_extractor = LandmarkExtractor()

    # Store landmarks from previous frame for interpolation
    previous_landmarks = None

    # Define sequence length in frames (90 frames = 3 seconds at 30 fps)
    sequence_length = 90
    # Counter for processed frames (unused variable)
    frames_counter = 0

    # List to store all processed data sequences
    data = []

    # Path to ADL sequences dataset
    dataset_path = "~/Downloads/URFall/adl_sequences"
    # Iterate through each folder in dataset
    for folder in sorted(os.listdir(dataset_path)):
        # Process each experiment (in a separate folder)
        # Queue stores feature vectors for frames
        queue = []
        # List of body landmarks for each frame
        bl = []
        # Labels for each frame (0 for ADL, 1 for fall)
        labels = []
        # List of images for visualization
        images = []

        # Process each image file in the folder
        for image in sorted(os.listdir(os.path.join(dataset_path, folder))):
            # Read image from disk
            image = cv2.imread(os.path.join(dataset_path, folder, image))

            # Extract body landmarks and aspect ratio from image
            body_landmarks, body_ar = landmark_extractor.get_body_landmarks(image)
            # If landmarks were successfully detected
            if body_landmarks:
                # Fix wrist landmarks that may be missed by detector
                body_landmarks = fix_wrist_landmarks(body_landmarks)

                # Validate that all landmarks are within image bounds
                check, keys = check_body_landmarks(body_landmarks)
                if not check:
                    print(f"{keys} landmarks are out of image!")
                # Store current landmarks for use if next frame fails detection
                previous_landmarks = body_landmarks

            else:
                # Use previous frame landmarks if detection fails
                body_landmarks = previous_landmarks
            
            # If landmarks are available, process them
            if body_landmarks is not None:
                # Convert landmarks to feature vector
                feature_vector = vectorize_landmarks(body_landmarks, body_ar)
                # Add feature vector to queue
                queue.append(feature_vector)
                # Store landmarks for later visualization
                bl.append(body_landmarks)
                # Label as ADL activity (0)
                labels.append(0)
                # Store image for visualization
                images.append(image)
        
        # At the end of a single experiment, check all extracted sequences and decide if consistent or not
        # Extract sequences with stride of 15 frames
        for i in range(0, len(queue) - sequence_length + 1, int(15)):
            print("New sequence...")
            print(f"Going from {i} to {i+sequence_length}")
            # List to store single sequence with label
            row = []

            # Build sequence of 90 frames
            for j in range(sequence_length):
                # Add feature vector to sequence
                row.append(queue[i + j])

                # Get image for current frame
                image = images[i+j]

                # Get landmarks for current frame
                body_landmarks = bl[i + j]
                # Convert normalized landmarks to image coordinates
                normalized_bl = landmark_extractor.de_normalize_body_landmarks(body_landmarks)
                # Draw landmarks on image
                out_img = landmark_extractor.draw_selected_landmarks(normalized_bl, image)
                # Display image with landmarks
                cv2.imshow('', out_img)
                # Wait 20ms between frames
                cv2.waitKey(20)
            
            # Add label to sequence
            row.append(labels[i+j])
            
            # Ask user if sequence should be saved
            ques = input("Save this sequence? \n")
            if ques == "y":
                # Add sequence to dataset
                data.append(row)

                print("Saved...")
                print(f"Dataset has now {len(data)} samples")
                print(f"Each sample has {len(data[-1])} elements")
                print(f"Each element has {len(data[-1][0])} elements")
                
            else:
                print("Discarded")

    # Save all processed data to JSON file
    dump_json(data) 



# Function to process fall videos from the dataset
def process_fall_videos():
    # Initialize landmark extractor
    landmark_extractor = LandmarkExtractor()

    # Store landmarks from previous frame for interpolation
    previous_landmarks = None

    # Define sequence length in frames (90 frames = 3 seconds at 30 fps)
    sequence_length = 90
    # Counter for processed frames (unused variable)
    frames_counter = 0

    # List to store all processed data sequences
    data = []

    # Path to fall sequences dataset
    dataset_path = "~/Downloads/URFall"
    # Iterate through each folder in dataset
    for folder in sorted(os.listdir(dataset_path)):
        # Process each experiment (in a separate folder)
        # Queue stores feature vectors for frames
        queue = []
        # List of body landmarks for each frame
        bl = []
        # Labels for each frame (0 for ADL, 1 for fall)
        labels = []
        # List of images for visualization
        images = []


        # Process each image file in the folder
        for image in sorted(os.listdir(os.path.join(dataset_path, folder))):
            # Read image from disk
            image = cv2.imread(os.path.join(dataset_path, folder, image))

            # Extract body landmarks and aspect ratio from image
            body_landmarks, body_ar = landmark_extractor.get_body_landmarks(image)
            # If landmarks were successfully detected
            if body_landmarks:
                # Fix wrist landmarks that may be missed by detector
                body_landmarks = fix_wrist_landmarks(body_landmarks)

                # Validate that all landmarks are within image bounds
                check, keys = check_body_landmarks(body_landmarks)
                if not check:
                    print(f"{keys} landmarks are out of image!")
                # Store current landmarks for use if next frame fails detection
                previous_landmarks = body_landmarks

            else:
                # Use previous frame landmarks if detection fails
                body_landmarks = previous_landmarks
            
            # Convert landmarks to feature vector
            feature_vector = vectorize_landmarks(body_landmarks, body_ar)
            # Add feature vector to queue
            queue.append(feature_vector)
            # Store landmarks for later visualization
            bl.append(body_landmarks)
            # Label as fall activity (1)
            labels.append(1)
            # Store image for visualization
            images.append(image)
        
        # At the end of a single experiment, check all extracted sequence and decide if consistent or not
        # Extract sequences with stride of 15 frames
        for i in range(0, len(queue) - sequence_length + 1, int(15)):
            print("New sequence...")
            print(f"Going from {i} to {i+sequence_length}")
            # List to store single sequence with label
            row = []

            # Build sequence of 90 frames
            for j in range(sequence_length):
                # Add feature vector to sequence
                row.append(queue[i + j])

                # Get image for current frame
                image = images[i+j]

                # Get landmarks for current frame
                body_landmarks = bl[i + j]
                # Convert normalized landmarks to image coordinates
                normalized_bl = landmark_extractor.de_normalize_body_landmarks(body_landmarks)
                # Draw landmarks on image
                out_img = landmark_extractor.draw_selected_landmarks(normalized_bl, image)
                # Display image with landmarks
                cv2.imshow('', out_img)
                # Wait 20ms between frames
                cv2.waitKey(20)
            
            # Add label to sequence
            row.append(labels[i+j])
            
            # Ask user if sequence should be saved
            ques = input("Save this sequence? \n")
            if ques == "y":
                # Add sequence to dataset
                data.append(row)

                print("Saved...")
                print(f"Dataset has now {len(data)} samples")
                print(f"Each sample has {len(data[-1])} elements")
                print(f"Each element has {len(data[-1][0])} elements")
                
            else:
                print("Discarded")

    # Save all processed data to JSON file
    dump_json(data)       



# Function to save processed data to JSON file
def dump_json(data):
    # Output file path
    out_path = "~/WatchDog/Perception/Dataset/ur_fall_FALL.json"

    # Open file for writing
    with open(out_path, 'w') as json_file:
        json_file.write('[\n')
        # Iterate through each sequence in dataset
        for row_idx, row in enumerate(data):
            json_file.write('  [\n')
            # Iterate through each element (feature vector or label) in sequence
            for vector_idx, vector in enumerate(row):
                # Convert vector to JSON string with indentation
                vector_str = '    ' + json.dumps(vector)
                # Add comma after all elements except the last
                if vector_idx < len(row) - 1:
                    json_file.write(f'{vector_str},\n')
                else:
                    json_file.write(f'{vector_str}\n')
            # Add comma after all sequences except the last
            if row_idx < len(data) - 1:
                json_file.write('  ],\n')
            else:
                json_file.write('  ]\n')
        json_file.write(']\n')



# Function to convert landmark coordinates to feature vector
def vectorize_landmarks(landmarks, body_ar):
    # Initialize empty vector
    input_vector = []
    # Iterate through each landmark key
    for key in landmarks.keys():
        # Add x coordinate of landmark
        input_vector.append(float(landmarks[key][0]))
        # Add y coordinate of landmark
        input_vector.append(float(landmarks[key][1]))

    # Add body aspect ratio as final feature
    input_vector.append(body_ar)
    # Return feature vector
    return input_vector
    

# Function to fix missing wrist landmarks
def fix_wrist_landmarks(landmarks):
    '''
    Hand landmark detector does not capture wrist landmarks, in this case the best thing is to assume person falling with
    straight arms pointing the floor, so wrist landmarks can be similar to hip landmarks
    '''
    # Generate random offset for x coordinate
    rand_x = random.uniform(0.001, 0.01)
    # Generate random offset for y coordinate
    rand_y = random.uniform(0.001, 0.01)

    # Check if left wrist was not detected
    if landmarks['left_wrist'] == [0, 0]:
        # Set left wrist to left hip position with random offset
        landmarks['left_wrist'][0] = landmarks['left_hip'][0] + rand_x
        landmarks['left_wrist'][1] = landmarks['left_hip'][1] + rand_y
    # Check if right wrist was not detected
    if landmarks['right_wrist'] == [0, 0]:
        # Set right wrist to right hip position with random offset
        landmarks['right_wrist'][0] = landmarks['right_hip'][0] + rand_x
        landmarks['right_wrist'][1] = landmarks['right_hip'][1] + rand_y
    # Return corrected landmarks
    return landmarks


# Function to validate body landmarks are within image bounds
def check_body_landmarks(coords):
    # Iterate through each landmark
    for key in coords.keys():
        # Iterate through x,y coordinates
        for landmark in coords[key]:
            # Check if coordinate is None
            if coords[key][0] is None:
                # Return False and the problematic key
                return False, key
            # Check if landmark is at origin (0,0) indicating detection failure
            elif coords[key][0] == float(0) and coords[key][1] == float(0):
                # Return False and the problematic key
                return False, key
    # All landmarks are valid
    return True, "None"


# Main execution block
if __name__ == "__main__":
    # Process fall videos from dataset
    process_fall_videos()