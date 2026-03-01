# Import the os module for file and directory operations
import os
# Import cv2 (OpenCV) for image processing and display
import cv2
# Import json for JSON file handling
import json
# Import random for generating random numbers
import random
# Import numpy for numerical operations
import numpy as np
# Import mediapipe for pose and landmark detection
import mediapipe as mp

# Add parent directory to system path for custom module imports
import sys
sys.path.append("../")
# Import custom LandmarkExtractor class for extracting body landmarks
from landmark_extractor import LandmarkExtractor

'''
UP-FALL Dataset structure
citation: https://sites.google.com/up.edu.mx/har-up/
17-folders, one per subject:
    11-subfolders, one per activity:
        3 - subfoldes, one per trial:
            1-CSV file containing the sensor signals of the attempt.

Classify falling activities as '1'
Classify all other activities as '0' - non falling - we have no activities where a person is standing

Use: 
17 subject-related data
5 activities
2 trials 
1 camera per trial

Read from each sigle folder, get landmarks and wrap into temporal sequences of 3 seconds
'''


def main():
    # Initialize the landmark extractor
    landmark_extractor = LandmarkExtractor()

    # Store landmarks from previous frame for fallback if current frame has no detections
    previous_landmarks = None

    # Define sequence length (90 frames for 3-second sequences at 30 fps)
    sequence_length = 90

    # Initialize counter for number of frames processed
    frames_counter = 0

    # List to store all processed sequences
    data = []

    # Path to the UP-Fall dataset
    dataset_path = "~/Downloads/UPFallDataset"
    
    # Iterate through each subject folder in the dataset
    for folder in sorted(os.listdir(dataset_path)):
        # Process each experiment (in a separate folder)
        
        # Queue to store feature vectors of the current sequence
        queue = []
        # List to store body landmarks for each frame
        bl = []
        # List to store labels (1 for fall, 0 for non-fall)
        labels = []
        # List to store image frames
        images = []

        # Iterate through each image file in the folder
        for image in sorted(os.listdir(os.path.join(dataset_path, folder))):
            # Read image from disk
            image = cv2.imread(os.path.join(dataset_path, folder, image))

            # Extract body landmarks and aspect ratio from the image
            body_landmarks, body_ar = landmark_extractor.get_body_landmarks(image)
            
            # If landmarks were detected
            if body_landmarks:
                # Fix missing wrist landmarks
                body_landmarks = fix_wrist_landmarks(body_landmarks)

                # Check if all landmarks are within image boundaries
                check, keys = check_body_landmarks(body_landmarks)
                if not check:
                    print(f"{keys} landmarks are out of image!")
                # Update previous landmarks for next frame
                previous_landmarks = body_landmarks

            # If no landmarks detected, use landmarks from previous frame
            else:
                body_landmarks = previous_landmarks
            
            # Convert landmarks to a feature vector
            feature_vector = vectorize_landmarks(body_landmarks, body_ar)
            # Add feature vector to queue
            queue.append(feature_vector)
            # Store body landmarks
            bl.append(body_landmarks)
            # Store label (hardcoded as 1, should be determined from dataset)
            labels.append(1)
            # Store original image
            images.append(image)
        
        # At the end of a single experiment, check all extracted sequences and decide if consistent or not
        # Process sequences with stride of 15 frames to create overlapping sequences
        for i in range(0, len(queue) - sequence_length + 1, int(15)):
            print("New sequence...")
            print(f"Going from {i} to {i+sequence_length}")
            # List to store a single sequence
            row = []

            # Iterate through each frame in the sequence
            for j in range(sequence_length):
                # Add feature vector to sequence
                row.append(queue[i + j])

                # Get the original image
                image = images[i+j]

                # Get body landmarks for this frame
                body_landmarks = bl[i + j]
                # Denormalize landmarks back to image coordinates
                normalized_bl = landmark_extractor.de_normalize_body_landmarks(body_landmarks)
                # Draw landmarks on the image
                out_img = landmark_extractor.draw_selected_landmarks(normalized_bl, image)
                # Display the annotated image
                cv2.imshow('', out_img)
                # Wait 20ms before showing next frame
                cv2.waitKey(20)
            
            # Append label to sequence
            row.append(labels[i+j])
            
            # Ask user if the sequence should be saved
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

    # Save all collected data to JSON file
    dump_json(data)       



def dump_json(data):
    # Define output file path
    out_path = "~/WatchDog/Perception/Dataset/fall-up.json"

    # Open file and write data in JSON format with custom formatting
    with open(out_path, 'w') as json_file:
        json_file.write('[\n')
        # Iterate through each row (sequence) in the data
        for row_idx, row in enumerate(data):
            json_file.write('  [\n')
            # Iterate through each element (feature vector or label) in the sequence
            for vector_idx, vector in enumerate(row):
                # Convert vector to JSON string with indentation
                vector_str = '    ' + json.dumps(vector)
                # Add comma if not the last element
                if vector_idx < len(row) - 1:
                    json_file.write(f'{vector_str},\n')
                else:
                    json_file.write(f'{vector_str}\n')
            # Add comma if not the last sequence
            if row_idx < len(data) - 1:
                json_file.write('  ],\n')
            else:
                json_file.write('  ]\n')
        json_file.write(']\n')



def vectorize_landmarks(landmarks, body_ar):
    # Initialize empty input vector
    input_vector = []
    # Iterate through each landmark
    for key in landmarks.keys():
        # Add x coordinate of landmark
        input_vector.append(float(landmarks[key][0]))
        # Add y coordinate of landmark
        input_vector.append(float(landmarks[key][1]))

    # Append body aspect ratio to feature vector
    input_vector.append(body_ar)
    # Return the complete feature vector
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

    # If left wrist is not detected (0, 0), approximate it using left hip position
    if landmarks['left_wrist'] == [0, 0]:
        # Set left wrist x coordinate relative to left hip
        landmarks['left_wrist'][0] = landmarks['left_hip'][0] + rand_x
        # Set left wrist y coordinate relative to left hip
        landmarks['left_wrist'][1] = landmarks['left_hip'][1] + rand_y
    # If right wrist is not detected (0, 0), approximate it using right hip position
    if landmarks['right_wrist'] == [0, 0]:
        # Set right wrist x coordinate relative to right hip
        landmarks['right_wrist'][0] = landmarks['right_hip'][0] + rand_x
        # Set right wrist y coordinate relative to right hip
        landmarks['right_wrist'][1] = landmarks['right_hip'][1] + rand_y
    # Return corrected landmarks
    return landmarks


def check_body_landmarks(coords):
    # Iterate through each landmark key
    for key in coords.keys():
        # Iterate through each coordinate (x, y)
        for landmark in coords[key]:
            # Check if x coordinate is None
            if coords[key][0] is None:
                # Return False and the problematic key
                return False, key
            # Check if both coordinates are zero (not detected)
            elif coords[key][0] == float(0) and coords[key][1] == float(0):
                # Return False and the problematic key
                return False, key
    # If all checks pass, return True
    return True, "None"


# Execute main function when script is run directly
if __name__ == "__main__":
    main()