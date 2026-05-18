# Import operating system utilities for file and directory operations
import os
# Import OpenCV library for computer vision tasks
import cv2
# Import JSON for handling JSON file serialization
import json
# Import random for generating random numbers
import random
# Import NumPy for numerical computations
import numpy as np
# Import MediaPipe for pose and hand landmark detection
import mediapipe as mp

# Add parent directory to Python path for importing custom modules
import sys
sys.path.append("../")
# Import custom LandmarkExtractor class for extracting body landmarks
from landmark_extractor import LandmarkExtractor

'''
Le2i Dataset structure
citation: https://www.kaggle.com/datasets/tuyenldvn/falldataset-imvia
6 folders, each one  with a falling scenario
Label all activities as '1' or '0' after checking

Read from each sigle folder, there is another nested folder with the 'Videos' folder get landmarks and wrap into temporal sequences of 3 seconds
'''


def process_dataset():
    """
    Main function to process fall simulation dataset.
    Extracts body landmarks from videos, creates temporal sequences, and saves them to JSON.
    """
    # Initialize landmark extractor for pose detection
    landmark_extractor = LandmarkExtractor()

    # Store landmarks from previous frame for continuity
    previous_landmarks = None

    # Define sequence length for temporal chunks (90 frames)
    sequence_length = 90

    # Initialize list to store processed data
    data = []
    # Path to dataset directory
    dataset_path = "~/Downloads/High_quality_fall/Fall_Simulation_Data/1"

    # Iterate through each video sequence in the dataset
    for video_sequence in sorted(os.listdir(dataset_path)):
        print(f"Processing sequence {video_sequence}")

        # Reset landmarks and buffers for new sequence
        previous_landmarks = None
        # Queue to store feature vectors for current video
        queue = []
        # Buffer to store body landmarks for current video
        bl = []
        # List to store labels for each frame
        labels = []
        # List to store frames for visualization
        images = []
        # Counter for processed frames
        frames_counter = 1

        # Open video file and read frames
        video_cap = cv2.VideoCapture(os.path.join(dataset_path, video_sequence))
        ret, image = video_cap.read()

        # Process each frame in the video
        while ret:

            # Extract body landmarks and aspect ratio from frame
            body_landmarks, body_ar = landmark_extractor.get_body_landmarks(image)
            # If landmarks detected, process and store them
            if body_landmarks:
                # Fix missing wrist landmarks with estimated values
                body_landmarks = fix_wrist_landmarks(body_landmarks)

                # Validate that all landmarks are within image bounds
                check, keys = check_body_landmarks(body_landmarks)
                if not check:
                    print(f"{keys} landmarks are out of image!")
                # Update previous landmarks for next frame
                previous_landmarks = body_landmarks

            else:
                # Use previous frame landmarks if detection fails
                body_landmarks = previous_landmarks
            
            # If valid landmarks available, create feature vector and store
            if body_landmarks is not None:
                # Convert landmarks to feature vector
                feature_vector = vectorize_landmarks(body_landmarks, body_ar)
                # Append vector to queue
                queue.append(feature_vector)
                # Store landmarks for visualization
                bl.append(body_landmarks)
                # Mark as falling activity (label 1)
                labels.append(1)
                # Store original frame
                images.append(image)

            # Read next frame
            ret, image = video_cap.read()
            # Increment frame counter
            frames_counter += 1

            # Stop processing after 2000 frames
            if frames_counter == 2000:
                break

        # Initialize skip interval for sequence sampling
        skip = int(15)

        # Increase skip interval based on total frames to reduce dataset size
        if len(queue) > 200:
            skip = int(30)

        # Further increase skip interval for very long videos
        if len(queue) > 1000:
            skip = int(60)

        # Create temporal sequences of fixed length with specified skip interval
        for i in range(0, len(queue) - sequence_length + 1, skip):
            print("New sequence...")
            print(f"Going from {i} to {i+sequence_length}")
            # Initialize row to store current sequence
            row = []

            # Collect sequence_length feature vectors
            for j in range(sequence_length):
                # Append feature vector to sequence
                row.append(queue[i + j])

                # Get original frame for visualization
                image = images[i+j]

                # Get body landmarks for current frame
                body_landmarks = bl[i + j]
                # De-normalize landmarks to pixel coordinates
                normalized_bl = landmark_extractor.de_normalize_body_landmarks(body_landmarks)
                # Draw landmarks on image
                out_img = landmark_extractor.draw_selected_landmarks(normalized_bl, image)
                # Display frame with landmarks
                cv2.imshow('', out_img)
                # Wait 20ms between frames
                cv2.waitKey(20)
            
            # Prompt user to validate sequence
            ques = input("Save this sequence? \n")
            # User selects label 0 (non-fall)
            if ques == "0":
                # Append label 0 to sequence
                row.append(0)
                # Store sequence in dataset
                data.append(row)

                print("Saved...")
                print(f"Dataset has now {len(data)} samples")
            
            # User selects label 1 (fall)
            elif ques == "1":
                # Append label 1 to sequence
                row.append(1)
                # Store sequence in dataset
                data.append(row)

                print("Saved...")
                print(f"Dataset has now {len(data)} samples")
            
            # User skips to next video
            elif ques == "s":
                break

            # User rejects sequence
            else:
                print("Discarded")

    # Save processed dataset to JSON file
    dump_json(data)       



def dump_json(data):
    """
    Save dataset to JSON file in nested array format.
    Args:
        data: List of sequences, each containing feature vectors and a label
    """
    # Define output file path
    out_path = "~/WatchDog/Perception/Dataset/Data_Files/fall_simulation_dataset.json"

    # Open file for writing
    with open(out_path, 'w') as json_file:
        # Write opening bracket for JSON array
        json_file.write('[\n')
        # Iterate through each sequence in dataset
        for row_idx, row in enumerate(data):
            # Write opening bracket for sequence
            json_file.write('  [\n')
            # Iterate through each element (vectors and label) in sequence
            for vector_idx, vector in enumerate(row):
                # Convert element to JSON string with indentation
                vector_str = '    ' + json.dumps(vector)
                # Add comma if not last element
                if vector_idx < len(row) - 1:
                    json_file.write(f'{vector_str},\n')
                else:
                    json_file.write(f'{vector_str}\n')
            # Add comma after sequence if not last sequence
            if row_idx < len(data) - 1:
                json_file.write('  ],\n')
            else:
                json_file.write('  ]\n')
        # Write closing bracket for JSON array
        json_file.write(']\n')



def vectorize_landmarks(landmarks, body_ar):
    """
    Convert landmark dictionary to flat feature vector.
    Args:
        landmarks: Dictionary mapping landmark names to [x, y] coordinates
        body_ar: Body aspect ratio value
    Returns:
        List containing x, y coordinates for all landmarks plus body aspect ratio
    """
    # Initialize input vector
    input_vector = []
    # Iterate through each landmark key
    for key in landmarks.keys():
        # Append x coordinate as float
        input_vector.append(float(landmarks[key][0]))
        # Append y coordinate as float
        input_vector.append(float(landmarks[key][1]))

    # Append body aspect ratio to vector
    input_vector.append(body_ar)
    # Return feature vector
    return input_vector
    

def fix_wrist_landmarks(landmarks):
    '''
    Hand landmark detector does not capture wrist landmarks, in this case the best thing is to assume person falling with
    straight arms pointing the floor, so wrist landmarks can be similar to hip landmarks
    '''
    # Generate random x offset
    rand_x = random.uniform(0.001, 0.01)
    # Generate random y offset
    rand_y = random.uniform(0.001, 0.01)

    # Check if left wrist landmark is missing
    if landmarks['left_wrist'] == [0, 0]:
        # Set left wrist x coordinate based on left hip with random offset
        landmarks['left_wrist'][0] = landmarks['left_hip'][0] + rand_x
        # Set left wrist y coordinate based on left hip with random offset
        landmarks['left_wrist'][1] = landmarks['left_hip'][1] + rand_y
    # Check if right wrist landmark is missing
    if landmarks['right_wrist'] == [0, 0]:
        # Set right wrist x coordinate based on right hip with random offset
        landmarks['right_wrist'][0] = landmarks['right_hip'][0] + rand_x
        # Set right wrist y coordinate based on right hip with random offset
        landmarks['right_wrist'][1] = landmarks['right_hip'][1] + rand_y
    # Return landmarks with fixed wrist positions
    return landmarks


def check_body_landmarks(coords):
    """
    Validate that all body landmarks have valid coordinates within image bounds.
    Args:
        coords: Dictionary mapping landmark names to [x, y] coordinates
    Returns:
        Tuple of (is_valid: bool, invalid_landmark_key: str)
    """
    # Iterate through each landmark in coordinates dictionary
    for key in coords.keys():
        # Iterate through each coordinate value
        for landmark in coords[key]:
            # Check if x coordinate is None (not detected)
            if coords[key][0] is None:
                # Return False and the invalid landmark key
                return False, key
            # Check if coordinates are [0, 0] (invalid/undetected landmark)
            elif coords[key][0] == float(0) and coords[key][1] == float(0):
                # Return False and the invalid landmark key
                return False, key
    # Return True if all landmarks are valid
    return True, "None"


# Check if script is run directly (not imported)
if __name__ == "__main__":
    # Execute main dataset processing function
    process_dataset()