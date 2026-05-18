# Import OpenCV for computer vision operations
import cv2
# Import json for handling JSON file operations
import json
# Import random for generating random numbers
import random
# Import NumPy for numerical operations
import numpy as np
# Import MediaPipe for pose/landmark detection
import mediapipe as mp
# Import custom LandmarkExtractor class for extracting body landmarks
from landmark_extractor import LandmarkExtractor


# DatasetCreator class: Creates a dataset of body landmarks from video frames
class DatasetCreator:
    # Constructor: Initializes the dataset creator with camera and landmark extractor
    def __init__(self):
        # Initialize video capture from default camera (index 0)
        self.camera_channel = cv2.VideoCapture(0)
        # Create instance of LandmarkExtractor to detect body landmarks
        self.landmark_extractor = LandmarkExtractor()
        # Define keys for the 11 body landmarks to be tracked
        self.landmarks_keys = ['front_face','left_wrist','right_wrist','left_shoulder','right_shoulder','left_hip','right_hip','left_knee','right_knee','right_ankle','left_ankle']

        # Get fps to compute how many frames to check to cover a video sequence of 5 seconds
        self.fps = self.camera_channel.get(cv2.CAP_PROP_FPS)
        # Set coverage duration in seconds (time window for sequence)
        self.coverage = 3
        # Calculate sequence length based on fps and coverage time
        self.sequence_length = int(self.fps * self.coverage)

        # Initialize queue to store feature vectors (landmarks)
        self.queue = []
        # Initialize list to store labels for each frame
        self.labels = []

        # Initialize frame counter
        self.frames_counter = 0 # Now just register 10000 frames for a fake dataset to create the model
        # Set output path for the dataset JSON file
        self.out_path = "~/WatchDog/Perception/Dataset/dataset.json"
        # Initialize queue to store frames (for visualization/validation)
        self.images_queue = []


    # Method: Main loop to capture frames and extract landmarks
    def start(self):
        # Loop while camera is open
        while(self.camera_channel.isOpened()):
            # Read frame from camera
            _, frame = self.camera_channel.read()

            # Store image dimensions [width, height]
            self.image_size = [frame.shape[1], frame.shape[0]]

            # Extract body landmarks and body aspect ratio from frame
            body_landmarks, body_ar = self.landmark_extractor.get_body_landmarks(frame)

            # self.display_landmarks(body_landmarks, frame)
            # Check if landmarks were detected
            if body_landmarks:
                # Validate that landmarks are within expected bounds
                if self.check_landmarks(body_landmarks):
                    # Convert landmarks to feature vector
                    feature_vector = self.vectorize_landmarks(body_landmarks, body_ar)
                    # Add feature vector to queue
                    self.queue.append(feature_vector)
                    # Add random label (0 or 1) for training
                    self.labels.append(random.randint(0,1))
                    # Store original frame for later validation
                    self.images_queue.append(frame)

                    # Increment processed frame counter
                    self.frames_counter += 1
                    # Print progress
                    print(f"Processed frame {self.frames_counter}/100....")
                    # Stop after processing 1000 frames
                    if self.frames_counter == 1000:
                        break

        # Save collected data to JSON file
        self.dump_dataset()
        # self.check()


    # Method: Validate collected data by drawing landmarks on frames
    def check(self):
        # Load dataset from JSON file
        with open(self.out_path, 'r') as json_file:
            data = json.load(json_file)

        # Initialize counter
        i = 0
        # Get first sample from dataset
        sample = data[0]
        # Iterate through each feature vector in sample
        for feature_vector in sample:
            # Copy frame for drawing
            out_img = self.images_queue[i].copy()
            i = i+1

            # Extract only landmark coordinates (ignore body aspect ratio)
            feature_vector = feature_vector[:22] # ignore body aspect ratio
            # Draw circles at landmark positions
            for c in range(0, len(feature_vector), 2):
                # Denormalize x coordinate to image width
                feature_vector[c] = int(feature_vector[c]*self.image_size[0])
                # Denormalize y coordinate to image height
                feature_vector[c+1] = int(feature_vector[c+1]*self.image_size[1])

                # Draw red circle at landmark position
                cv2.circle(out_img, (feature_vector[c], feature_vector[c+1]), 4, (0, 0,255), thickness=-1)

            # Display image and wait for key press
            cv2.imshow('', out_img)
            cv2.waitKey(0)
            
                
    # Method: Export collected data to JSON file        
    def dump_dataset(self):
        '''
        Export to json file
        PyTorch DataLoader will load data as an array where each element is a sample related to a frame and process it frame by frame
        Each sample has 23 features (11 x,y body landmarks + body aspect ratio) and a label
        Shift window by a factor of 15 frames
        '''
        # Initialize data list to store all sequences
        data = []        

        # Write data rows
        # Iterate through frames with sliding window (step size = fps/2)
        for i in range(0, len(self.queue) - self.sequence_length + 1, int(self.fps/2)):
            print(f"writing strting from i {i}")
            # Create a sequence of feature vectors
            row = []
            # Collect sequence_length consecutive feature vectors
            for j in range(self.sequence_length):
                row.append(self.queue[i + j])
            
            print(f"Writing from {i} to {i+j}")
            
            # Add label for this sequence
            if self.labels[i+j] is not None:
                row.append(self.labels[i+j])
            
            # Add completed sequence to data
            data.append(row)

        # Write data to JSON file with formatted structure
        with open(self.out_path, 'w') as json_file:
            json_file.write('[\n')
            # Iterate through each sequence (row)
            for row_idx, row in enumerate(data):
                json_file.write('  [\n')
                # Iterate through each feature vector in sequence
                for vector_idx, vector in enumerate(row):
                    # Convert vector to JSON string with indentation
                    vector_str = '    ' + json.dumps(vector)
                    # Add comma if not last vector
                    if vector_idx < len(row) - 1:
                        json_file.write(f'{vector_str},\n')
                    else:
                        json_file.write(f'{vector_str}\n')
                # Add comma if not last sequence
                if row_idx < len(data) - 1:
                    json_file.write('  ],\n')
                else:
                    json_file.write('  ]\n')
            json_file.write(']\n')

            
    # Method: Convert landmarks dictionary to feature vector
    def vectorize_landmarks(self, landmarks, body_ar):
        # Initialize empty input vector
        input_vector = []
        # Iterate through each landmark key
        for key in self.landmarks_keys:
            # Append x coordinate of landmark
            input_vector.append(float(landmarks[key][0]))
            # Append y coordinate of landmark
            input_vector.append(float(landmarks[key][1]))

        # Append body aspect ratio as final feature
        input_vector.append(body_ar)
        # Return complete feature vector
        return input_vector


    # Method: Display landmarks on image for visualization
    def display_landmarks(self, body_landmarks, image):
        # Denormalize landmarks from [0,1] range to image coordinates
        normal_bl = self.landmark_extractor.de_normalize_body_landmarks(body_landmarks)
        # Draw landmarks on image
        out_img = self.landmark_extractor.draw_selected_landmarks(normal_bl, image)
        # Display image
        cv2.imshow('', out_img)
        # Wait for key press (20ms timeout, exit on ESC key)
        cv2.waitKey(20) == 27
            

    # Method: Validate that landmarks are within normalized bounds [0,1]
    def check_landmarks(self, body_landmarks):
        # Iterate through each landmark
        for key in body_landmarks:
            # Check if coordinates are outside valid range [0,1]
            if(body_landmarks[key][0]<=0 or body_landmarks[key][0]>1 or body_landmarks[key][1]<=0 or body_landmarks[key][1]>1):
                # Return False if any landmark is invalid
                return False
        # Return True if all landmarks are valid
        return True


# Entry point: Create DatasetCreator instance and start processing
if __name__ == "__main__":
    # Initialize dataset creator
    dataset_creator = DatasetCreator()
    # Start capturing and processing frames
    dataset_creator.start()