# Import OpenCV for image processing and video capture
import cv2
# Import sys to manipulate Python path
import sys
# Import PyTorch for deep learning models
import torch
# Import random for generating random numbers
import random
# Import torch.nn for neural network layers
import torch.nn as nn
# Import threading Lock for thread-safe operations
from threading import Lock
# Import torch.optim for optimization (unused in this file)
import torch.optim as optim
# Import custom model class for fall detection
from model import BodyClassificationNetwork

# Add parent dataset directory to Python path
sys.path.append("../Dataset")
# Import custom landmark extraction module
from landmark_extractor import LandmarkExtractor


# Class for detecting falls using a transformer-based neural network
class FallDetectionTransformer():
    
    def __init__(self):
        # Initialize landmark extractor for body pose estimation
        self.landmark_extractor = LandmarkExtractor()
        
        # Set device to GPU if available, otherwise use CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Configure transformer model hyperparameters
        options = {'num_transformer_blocks':2,
               'd_input':23,  # Input dimension (23 landmarks × 2 coordinates)
               'd_model':32,  # Model dimension
               'num_heads':4,  # Number of attention heads
               'num_layers':2,  # Transformer layers
               'd_ff':1024,  # Feed-forward dimension
               'max_seq_length':90,  # Maximum sequence length (3 seconds at 30 FPS)
               'dropout':0.1,  # Dropout rate
               'inner_df':64,  # Inner dimension
                }
        
        # Instantiate the transformer model with specified options
        self.model = BodyClassificationNetwork(options['num_transformer_blocks'], 
                                                options['d_input'], 
                                                options['d_model'], 
                                                options['num_heads'], 
                                                options['num_layers'], 
                                                options['d_ff'], 
                                                options['max_seq_length'], 
                                                options['dropout'], 
                                                options['inner_df'])
        
        # Load pre-trained model weights
        weights_path = "../runs/train/best_model.pth"
        self.model.load_state_dict(torch.load(weights_path))
        # Move model to GPU/CPU device
        self.model.to(self.device)
        # Set model to evaluation mode (disables dropout, etc.)
        self.model.eval()
        print("FallDetectionTransformer successfully LOADED...")

        # Initialize queue to store feature vectors for sliding window
        self.queue = []
        # Thread-safe lock for queue access
        self.queue_lock = Lock()
        # Current prediction state
        self.prediction = "Standing"

        
    # Main detection loop
    def run(self, source):
        # Open video source (0 for webcam, or path to video file)
        video_capture = cv2.VideoCapture(source)
        
        # Continuous video processing loop
        while True:
            # Read frame from video source
            _, frame = video_capture.read()

            # Extract body landmarks and aspect ratio from frame
            body_landmarks, body_ar = self.landmark_extractor.get_body_landmarks(frame)
            
            # Process only if landmarks are detected
            if body_landmarks:
                # Fix missing wrist landmarks with interpolated values
                body_landmarks = self.fix_wrist_landmarks(body_landmarks)

                # Validate landmark quality
                check, keys = self.check_body_landmarks(body_landmarks)
                if check:
                    # Convert landmarks to normalized feature vector
                    feature_vector = self.vectorize_landmarks(body_landmarks, body_ar)
                    
                    # Thread-safely append feature vector to queue
                    with self.queue_lock:
                        self.queue.append(feature_vector)
                    
                    # Process prediction when queue reaches 90 frames (3 seconds)
                    if len(self.queue) == 90:
                        # Convert queue to tensor and add batch dimension
                        input_vector = torch.Tensor(self.queue).to(self.device).unsqueeze(0)
                        
                        # Inference without gradient computation
                        with torch.no_grad():
                            # Get model prediction probability
                            prediction = self.model(input_vector).numpy()[0]
                            print(prediction)
                            
                            # Classify as fall or standing based on threshold
                            if prediction > 0.5:
                                self.prediction = "Fall"
                            else:
                                self.prediction = "Standing"

                        # Shift sliding window by 45 frames for next prediction
                        self.queue = self.queue[45:]
                
                    # Prepare output image with prediction text
                    out_image = self.get_out_image(frame.copy(), self.prediction)
                    # Denormalize landmarks to image coordinates
                    normalized_bl = self.landmark_extractor.de_normalize_body_landmarks(body_landmarks)
                    # Draw landmarks on output image
                    out_img = self.landmark_extractor.draw_selected_landmarks(normalized_bl, out_image)
                    # Display result window
                    cv2.imshow("Fall Detection", out_image)
                    
                    # Exit on ESC key press (27)
                    if cv2.waitKey(1) == 27:
                        break

    # Overlay prediction text on frame
    def get_out_image(self, frame, text):
        # Set text position (x, y) in pixels
        position = (50, 50)
        # Font type
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Font size
        font_scale = 1

        # Red color for fall, green for standing
        if text == "Fall":
            color = (0, 0, 255)  # BGR format: red
        else:
            color = (0, 255, 0)  # BGR format: green

        # Text line thickness
        thickness = 2

        # Write prediction text on frame
        cv2.putText(frame, text, position, font, font_scale, color, thickness)
        return frame

    # Convert landmarks dict to normalized feature vector
    def vectorize_landmarks(self, landmarks, body_ar):
        # Initialize empty feature vector
        input_vector = []
        
        # Iterate through all landmark coordinates
        for key in landmarks.keys():
            # Append x and y coordinates for each landmark
            input_vector.append(float(landmarks[key][0]))
            input_vector.append(float(landmarks[key][1]))

        # Define normalization range for body aspect ratio
        min_bar = 0.0
        max_bar = 40.0

        # Normalize body aspect ratio
        if body_ar > 40:
            body_ar = body_ar / 10  # Scale down large aspect ratios
        
        # Min-max normalization
        norm_body_ar = (body_ar - min_bar) / (max_bar - min_bar)

        # Append normalized aspect ratio to feature vector
        input_vector.append(norm_body_ar)
        return input_vector

    # Handle missing wrist landmarks by interpolating from hip position
    def fix_wrist_landmarks(self, landmarks):
        # Generate small random noise for plausible coordinates
        rand_x = random.uniform(0.001, 0.01)
        rand_y = random.uniform(0.001, 0.01)

        # Replace missing left wrist with perturbed left hip position
        if landmarks['left_wrist'] == [0, 0]:
            landmarks['left_wrist'][0] = landmarks['left_hip'][0] + rand_x
            landmarks['left_wrist'][1] = landmarks['left_hip'][1] + rand_y
            
        # Replace missing right wrist with perturbed right hip position
        if landmarks['right_wrist'] == [0, 0]:
            landmarks['right_wrist'][0] = landmarks['right_hip'][0] + rand_x
            landmarks['right_wrist'][1] = landmarks['right_hip'][1] + rand_y
        return landmarks
    
    # Validate landmark quality and completeness
    def check_body_landmarks(self, coords):
        # Check all landmarks for validity
        for key in coords.keys():
            # Verify x and y coordinates exist
            for landmark in coords[key]:
                # Return False if coordinate is None
                if coords[key][0] is None:
                    return False, key
                # Return False if coordinate is at origin (missing)
                elif coords[key][0] == float(0) and coords[key][1] == float(0):
                    return False, key
        # All landmarks valid
        return True, "None"


# Entry point: create detector and run on webcam
if __name__ == "__main__":
    # Instantiate fall detection system
    detector = FallDetectionTransformer()
    # Run detection on webcam (source=0)
    # Alternative: detector.run(source="~/Downloads/Le2i/Coffee_room_01/Coffee_room_01/Videos/video (1).avi")
    detector.run(source=0)