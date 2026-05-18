# Import operating system utilities for file and path operations
import os
# Import OpenCV library for computer vision and image processing
import cv2
# Import JSON library for reading/writing JSON files
import json
# Import random for generating random numbers
import random
# Import NumPy for numerical array operations
import numpy as np
# Import MediaPipe for pose/landmark detection
import mediapipe as mp

# Add parent directory to Python path for importing custom modules
import sys
sys.path.append("../")
# Import custom landmark extraction module
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
    Main function to process Le2i dataset videos, extract landmarks, and create temporal sequences.
    Allows manual labeling of falling scenarios and saves data to JSON file.
    """
    # Initialize landmark extractor for pose detection
    landmark_extractor = LandmarkExtractor()

    # Store landmarks from previous frame for handling missing detections
    previous_landmarks = None

    # Define temporal sequence length (90 frames = 3 seconds at 30 fps)
    sequence_length = 90
    # Counter for frames processed (unused variable)
    frames_counter = 0

    # List to store all labeled sequences
    data = []
    # Path to Le2i dataset
    dataset_path = "~/Downloads/Le2i/Office"

    # Iterate through each video sequence in the dataset directory
    for video_sequence in sorted(os.listdir(os.path.join(dataset_path, "Office"))):
        print(f"Processing sequence {video_sequence}")

        # Reset variables for each new video sequence
        previous_landmarks = None
        # Queue to store feature vectors for the current video
        queue = []
        # List to store body landmarks for each frame
        bl = []
        # List to store labels for each frame
        labels = []
        # List to store images for visualization
        images = []

        # Open video file using OpenCV
        video_cap = cv2.VideoCapture(os.path.join(dataset_path, "Office", video_sequence))
        # Read first frame from video
        ret, image = video_cap.read()

        # Process all frames in the video
        while ret:

            # Extract body landmarks and aspect ratio from current frame
            body_landmarks, body_ar = landmark_extractor.get_body_landmarks(image)
            # If landmarks detected, fix missing wrist landmarks
            if body_landmarks:
                body_landmarks = fix_wrist_landmarks(body_landmarks)

                # Check if all landmarks are within image bounds
                check, keys = check_body_landmarks(body_landmarks)
                if not check:
                    print(f"{keys} landmarks are out of image!")
                # Update previous landmarks for next frame
                previous_landmarks = body_landmarks

            else:
                # If no landmarks detected, use previous frame's landmarks
                body_landmarks = previous_landmarks
            
            # If landmarks are available, convert to feature vector and store
            if body_landmarks is not None:
                # Convert landmarks to normalized feature vector
                feature_vector = vectorize_landmarks(body_landmarks, body_ar)
                queue.append(feature_vector)
                bl.append(body_landmarks)
                labels.append(1)
                images.append(image)

            # Read next frame
            ret, image = video_cap.read()

        # Determine frame skip amount based on video length to balance dataset
        skip = int(15)

        # Skip more frames for longer videos to reduce redundancy
        if len(queue) > 200:
            skip = int(30)

        # Skip even more frames for very long videos
        if len(queue) > 1000:
            skip = int(60)

        # Create temporal sequences with specified length and skip interval
        for i in range(0, len(queue) - sequence_length + 1, skip):
            print("New sequence...")
            print(f"Going from {i} to {i+sequence_length}")
            # Initialize row to store current sequence
            row = []

            # Collect sequence_length consecutive frames
            for j in range(sequence_length):
                # Add feature vector to sequence
                row.append(queue[i + j])

                # Get original image for visualization
                image = images[i+j]

                # Get landmarks for current frame
                body_landmarks = bl[i + j]
                # Convert normalized landmarks back to pixel coordinates
                normalized_bl = landmark_extractor.de_normalize_body_landmarks(body_landmarks)
                # Draw landmarks on image
                out_img = landmark_extractor.draw_selected_landmarks(normalized_bl, image)
                # Display image with landmarks
                cv2.imshow('', out_img)
                cv2.waitKey(20)
            
            # Prompt user to label the sequence
            ques = input("Save this sequence? \n")
            # User input '0' - label as no fall
            if ques == "0":
                row.append(0)
                data.append(row)

                print("Saved...")
                print(f"Dataset has now {len(data)} samples")
            
            # User input '1' - label as fall
            elif ques == "1":
                row.append(1)
                data.append(row)

                print("Saved...")
                print(f"Dataset has now {len(data)} samples")
            
            # User input 's' - skip to next video
            elif ques == "s":
                break

            # Any other input - discard sequence
            else:
                print("Discarded")

    # Save all labeled sequences to JSON file
    dump_json(data)       



def dump_json(data):
    """
    Save labeled dataset to JSON file with formatted structure.
    
    Args:
        data: List of sequences, each containing feature vectors and a label
    """
    # Define output file path
    out_path = "~/WatchDog/Perception/Dataset/Data_Files/le2i_office.json"

    # Open output file and write data in JSON array format
    with open(out_path, 'w') as json_file:
        json_file.write('[\n')
        # Iterate through each sequence
        for row_idx, row in enumerate(data):
            json_file.write('  [\n')
            # Write each feature vector and label in the sequence
            for vector_idx, vector in enumerate(row):
                # Convert vector to JSON string with indentation
                vector_str = '    ' + json.dumps(vector)
                # Add comma if not last element
                if vector_idx < len(row) - 1:
                    json_file.write(f'{vector_str},\n')
                else:
                    json_file.write(f'{vector_str}\n')
            # Close sequence array with comma if not last row
            if row_idx < len(data) - 1:
                json_file.write('  ],\n')
            else:
                json_file.write('  ]\n')
        json_file.write(']\n')



def vectorize_landmarks(landmarks, body_ar):
    """
    Convert landmarks dictionary to flat feature vector with aspect ratio.
    
    Args:
        landmarks: Dictionary of body landmarks with x,y coordinates
        body_ar: Body aspect ratio value
        
    Returns:
        List containing flattened landmark coordinates followed by aspect ratio
    """
    # Initialize empty input vector
    input_vector = []
    # Iterate through all landmark keys
    for key in landmarks.keys():
        # Append x coordinate of landmark
        input_vector.append(float(landmarks[key][0]))
        # Append y coordinate of landmark
        input_vector.append(float(landmarks[key][1]))

    # Append body aspect ratio to end of vector
    input_vector.append(body_ar)
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

    # If left wrist not detected (0, 0), set to left hip position with noise
    if landmarks['left_wrist'] == [0, 0]:
        landmarks['left_wrist'][0] = landmarks['left_hip'][0] + rand_x
        landmarks['left_wrist'][1] = landmarks['left_hip'][1] + rand_y
    # If right wrist not detected (0, 0), set to right hip position with noise
    if landmarks['right_wrist'] == [0, 0]:
        landmarks['right_wrist'][0] = landmarks['right_hip'][0] + rand_x
        landmarks['right_wrist'][1] = landmarks['right_hip'][1] + rand_y
    return landmarks


def check_body_landmarks(coords):
    """
    Validate that all body landmarks are valid and within image bounds.
    
    Args:
        coords: Dictionary of body landmarks
        
    Returns:
        Tuple of (is_valid: bool, invalid_key: str)
    """
    # Check each landmark coordinate
    for key in coords.keys():
        # Iterate through x, y coordinates of each landmark (unused variable)
        for landmark in coords[key]:
            # Check if x coordinate is None
            if coords[key][0] is None:
                return False, key
            # Check if landmark is at origin (0, 0) - indicates missing detection
            elif coords[key][0] == float(0) and coords[key][1] == float(0):
                return False, key
    # All landmarks valid
    return True, "None"


# Main entry point - execute dataset processing when script is run
if __name__ == "__main__":
    process_dataset()