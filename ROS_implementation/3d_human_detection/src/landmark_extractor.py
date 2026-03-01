#!/usr/bin/python3.8
# Specify Python 3.8 interpreter

import cv2
# Import OpenCV library for image processing
import numpy as np
# Import NumPy for numerical operations
import mediapipe as mp
# Import MediaPipe for pose and hand landmark detection

class LandmarkExtractor():
    # Class to extract human body landmarks from images
    def __init__(self):
        # Initialize the extractor with MediaPipe solutions
        self.frame = None
        # Store the current frame
        self.image_size = None
        # Store the image dimensions [width, height]

        self.mp_drawing = mp.solutions.drawing_utils
        # Get MediaPipe drawing utilities
        self.mp_holistic = mp.solutions.holistic
        # Get MediaPipe holistic solution for pose, hand, and face detection
        self.holistic = self.mp_holistic.Holistic(min_detection_confidence=0.70)
        # Initialize holistic model with 70% confidence threshold


    def draw_pose_landmarks(self, landmarks, image):
        # Draw pose landmarks and connections on image
        self.mp_drawing.draw_landmarks(image, landmarks, self.mp_holistic.POSE_CONNECTIONS,
                                       self.mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2),
                                       self.mp_drawing.DrawingSpec(color=(180,100,100), thickness=2, circle_radius=2))
        # Draw landmarks in blue and connections in light red

    def draw_hand_landmarks(self, landmarks, image):
        # Draw hand landmarks and connections on image
        self.mp_drawing.draw_landmarks(image, landmarks, self.mp_holistic.HAND_CONNECTIONS,
                                       self.mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2),
                                       self.mp_drawing.DrawingSpec(color=(180,100,100), thickness=2, circle_radius=2))
        # Draw landmarks in blue and connections in light red
        
    def draw_selected_landmarks(self, landmarks, image):
        # Draw circles at specified landmark positions
        for key in landmarks.keys():
            # Iterate through each landmark
            cv2.circle(image, (landmarks[key][0], landmarks[key][1]), 10, (0, 255, 0), thickness=-1)
            # Draw green filled circle at landmark position

        return image
        # Return modified image
        
    
    def get_body_landmarks(self, image):
        # Extract body landmarks, wrist landmarks, and body aspect ratio from image
        self.image_size = [image.shape[1], image.shape[0]]
        # Store image dimensions (width, height)
        self.frame = image
        # Store the input frame

        results = self.holistic.process(image)
        # Process image with holistic model

        body_pose_landmarks = results.pose_landmarks
        # Extract pose landmarks
        right_hand_landmarks = results.right_hand_landmarks
        # Extract right hand landmarks
        left_hand_landmarks = results.left_hand_landmarks
        # Extract left hand landmarks

        normalized_landmarks = self.get_selected_body_landmarks(body_pose_landmarks)
        # Get selected normalized body landmarks

        if normalized_landmarks is not None:
            # If landmarks detected
            right_wrist = self.get_wrist_landmarks(right_hand_landmarks)
            # Get right wrist position
            left_wrist = self.get_wrist_landmarks(left_hand_landmarks)
            # Get left wrist position

            normalized_landmarks['left_wrist'] = left_wrist
            # Add left wrist to landmarks
            normalized_landmarks['right_wrist'] = right_wrist
            # Add right wrist to landmarks
            
            img_landmarks = self.de_normalize_body_landmarks(normalized_landmarks)
            # Convert normalized coordinates to image pixel coordinates
            vd, hd = self.get_body_aspect_ratio(img_landmarks)
            # Calculate vertical and horizontal distances

            if hd == 0:
                # If horizontal distance is zero
                hd = 1
                # Set to 1 to avoid division by zero
                print(f"Adjusting BAR {vd}/{hd}")
                # Print adjustment message
            body_aspect_ratio = vd/hd
            # Calculate body aspect ratio

            # image_out = image.copy()
            # Commented out: create copy of image
            # image_out = self.draw_selected_landmarks(img_landmarks, image)
            # Commented out: draw landmarks on image
            # image_out = self.check_body_aspect_ratio(vd, hd, img_landmarks)
            # Commented out: check and visualize aspect ratio
            # cv2.imshow('', image_out)
            # Commented out: display image
            # cv2.waitKey(20)
            # Commented out: wait 20ms for key press

            return img_landmarks, body_aspect_ratio
            # Return landmarks and aspect ratio
        
        else:
            # If no landmarks detected
            return None, None
            # Return None values
        
    
    def get_body_hip(self, image):
        # Extract hip landmark positions from image
        self.image_size = [image.shape[1], image.shape[0]]
        # Store image dimensions
        self.frame = image
        # Store the input frame

        results = self.holistic.process(image)
        # Process image with holistic model

        body_pose_landmarks = results.pose_landmarks
        # Extract pose landmarks

        normalized_landmarks = self.get_selected_body_landmarks(body_pose_landmarks)
        # Get selected normalized body landmarks

        if normalized_landmarks is not None:
            # If landmarks detected
            img_landmarks = self.de_normalize_body_landmarks(normalized_landmarks)
            # Convert normalized coordinates to image pixel coordinates

            return img_landmarks['left_hip'], img_landmarks['right_hip']
            # Return left and right hip positions
        
        else:
            # If no landmarks detected
            return None, None
            # Return None values


            
    def check_body_landmarks(self, coords):
        # Check and clamp landmark coordinates to valid range
        for key in coords.keys():
            # Iterate through each landmark
            for landmark in coords[key]:
                # Iterate through coordinate values
                if coords[key][0] > 1:
                    # If x-coordinate exceeds 1
                    coords[key][0] = 0
                    # Set to 0
                if coords[key][1] > 1:
                    # If y-coordinate exceeds 1
                    coords[key][1] = 0
                    # Set to 0


    def get_wrist_landmarks(self, hand_landmarks):
        # Extract wrist landmark (first point) from hand landmarks
        if hand_landmarks is not None:
            # If hand landmarks detected
            for idx, landmark in enumerate(hand_landmarks.landmark):
                # Iterate through landmarks
                if idx == 0:
                    # If first landmark (wrist)
                    return[landmark.x, landmark.y]
                    # Return wrist coordinates
                
        return [0, 0]
        # Return zero coordinates if no landmarks detected


    def get_selected_body_landmarks(self, body_landmarks):
        # Extract selected body landmarks from pose detection results
        coords = {'front_face':None, 'left_shoulder':None, 'right_shoulder':None, 'left_hip':None, 'right_hip':None, 'left_knee':None, 'right_knee':None, 'right_ankle':None, 'left_ankle':None}
        # Initialize dictionary with landmark names

        if body_landmarks is not None:
            # If body landmarks detected
            for idx, landmark in enumerate(body_landmarks.landmark):
                # Iterate through all landmarks
                if idx == 0:
                    # Nose/front face
                    coords['front_face'] = [landmark.x, landmark.y]
                elif idx == 11:
                    # Left shoulder
                    coords['left_shoulder'] = [landmark.x, landmark.y]
                elif idx == 12:
                    # Right shoulder
                    coords['right_shoulder'] = [landmark.x, landmark.y]
                elif idx == 23:
                    # Left hip
                    coords['left_hip'] = [landmark.x, landmark.y]
                elif idx == 24:
                    # Right hip
                    coords['right_hip'] = [landmark.x, landmark.y]
                elif idx == 25:
                    # Left knee
                    coords['left_knee'] = [landmark.x, landmark.y]
                elif idx == 26:
                    # Right knee
                    coords['right_knee'] = [landmark.x, landmark.y]
                elif idx == 28:
                    # Right ankle
                    coords['right_ankle'] = [landmark.x, landmark.y]
                elif idx == 27:
                    # Left ankle
                    coords['left_ankle'] = [landmark.x, landmark.y]

            return coords
            # Return dictionary with selected landmarks
        else: 
            # If no landmarks detected
            return None
            # Return None
        
    
    def de_normalize_body_landmarks(self, coords):
        # Convert normalized coordinates (0-1) to image pixel coordinates
        img_coords = {'front_face':None, 'left_shoulder':None, 'right_shoulder':None, 'left_hip':None, 'right_hip':None, 'left_knee':None, 'right_knee':None, 'right_ankle':None, 'left_ankle':None}
        # Initialize dictionary for pixel coordinates

        for key in coords.keys():
            # Iterate through each landmark
            img_coords[key] =  [int(coords[key][0]*self.image_size[0]), int(coords[key][1]*self.image_size[1])]
            # Multiply normalized coordinates by image dimensions and convert to integers

        return img_coords
        # Return dictionary with pixel coordinates
    

    def get_body_aspect_ratio(self, normalized_landmarks):
        # Calculate vertical and horizontal distance for body aspect ratio
        '''
        Here I compute vertical max distance between shoulders and knees, as well as their horizontal distance
        If the ratio vertical/horizontal is > 1 I am in a STANDING position, otherwise I am prone
        '''
        vertical_distance = max(abs(normalized_landmarks['left_shoulder'][1] - normalized_landmarks['right_knee'][1]), abs(normalized_landmarks['right_shoulder'][1] - normalized_landmarks['left_knee'][1]))
        # Calculate maximum vertical distance between shoulders and knees
        
        horizontal_distance = max(abs(normalized_landmarks['left_shoulder'][0] - normalized_landmarks['right_knee'][0]), abs(normalized_landmarks['right_shoulder'][0] - normalized_landmarks['left_knee'][0]))
        # Calculate maximum horizontal distance between shoulders and knees

        return vertical_distance, horizontal_distance
        # Return distances
    

    def check_body_aspect_ratio(self, vd, hd, img_landmarks):
        # Visualize body aspect ratio with lines on image
        image_out = self.frame.copy()
        # Create copy of frame for visualization
        center_shoulders = [int((img_landmarks['left_shoulder'][0] + img_landmarks['right_shoulder'][0]) //2), int(img_landmarks['left_shoulder'][1])]
        # Calculate center point between shoulders
        end_shoulders = [int((img_landmarks['left_shoulder'][0] + img_landmarks['right_shoulder'][0]) //2), int(img_landmarks['left_shoulder'][1])+vd]
        # Calculate end point for vertical line (vertical distance)
        end_body = [int(img_landmarks['right_shoulder'][0] + hd),  int((img_landmarks['right_shoulder'][1] + img_landmarks['right_knee'][1]) //2)]
        # Calculate end point for horizontal line (horizontal distance)
        center_body = [int(img_landmarks['right_shoulder'][0]), int((img_landmarks['right_shoulder'][1] + img_landmarks['right_knee'][1]) //2)]
        # Calculate center point of body

        print(center_shoulders)
        # Print shoulder center coordinates
        print(center_body)
        # Print body center coordinates
        cv2.line(image_out,  center_shoulders, end_shoulders, (0,0,255), 3)
        # Draw vertical red line from shoulders
        cv2.line(image_out, center_body, end_body, (0,0,255), 3)
        # Draw horizontal red line from body center

        return image_out
        # Return image with drawn lines
