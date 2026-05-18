import cv2
import numpy as np
import mediapipe as mp

class LandmarkExtractor():
    """
    Extracts human body landmarks (pose, hands) from images using MediaPipe Holistic.
    Provides methods to normalize/denormalize landmarks and compute body aspect ratio.
    """
    
    def __init__(self):
        """Initialize MediaPipe Holistic model and drawing utilities."""
        self.frame = None  # Stores the current frame being processed
        self.image_size = None  # Stores [width, height] of the current image

        self.mp_drawing = mp.solutions.drawing_utils  # MediaPipe drawing utilities
        self.mp_holistic = mp.solutions.holistic  # MediaPipe holistic solution
        # Initialize Holistic model with 70% confidence threshold for detection
        self.holistic = self.mp_holistic.Holistic(min_detection_confidence=0.70)

    def draw_pose_landmarks(self, landmarks, image):
        """Draw pose skeleton on image with customizable colors and thickness."""
        self.mp_drawing.draw_landmarks(image, landmarks, self.mp_holistic.POSE_CONNECTIONS,
                                       self.mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2),
                                       self.mp_drawing.DrawingSpec(color=(180,100,100), thickness=2, circle_radius=2))

    def draw_hand_landmarks(self, landmarks, image):
        """Draw hand skeleton on image with customizable colors and thickness."""
        self.mp_drawing.draw_landmarks(image, landmarks, self.mp_holistic.HAND_CONNECTIONS,
                                       self.mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2),
                                       self.mp_drawing.DrawingSpec(color=(180,100,100), thickness=2, circle_radius=2))
        
    def draw_selected_landmarks(self, landmarks, image):
        """Draw circles at specified landmark positions on the image."""
        for key in landmarks.keys():  # Iterate through each landmark
            # Draw filled circle at landmark position with red color
            cv2.circle(image, (landmarks[key][0], landmarks[key][1]), 4, (0, 0, 255), thickness=-1)

        return image  # Return modified image
        
    def get_body_landmarks(self, image):
        """
        Main method: extract normalized body landmarks from image.
        
        Returns:
            tuple: (normalized_landmarks dict, body_aspect_ratio) or (None, None) if no pose detected
        """
        self.image_size = [image.shape[1], image.shape[0]]  # Store image dimensions [width, height]
        self.frame = image  # Store frame for later use

        results = self.holistic.process(image)  # Run holistic detection on image

        # Extract landmark groups from results
        body_pose_landmarks = results.pose_landmarks
        right_hand_landmarks = results.right_hand_landmarks
        left_hand_landmarks = results.left_hand_landmarks

        # Get selected body landmarks (normalized to 0-1 range)
        normalized_landmarks = self.get_selected_body_landmarks(body_pose_landmarks)

        if normalized_landmarks is not None:  # If pose detected
            # Extract wrist positions from hand landmarks (normalized)
            right_wrist = self.get_wrist_landmarks(right_hand_landmarks)
            left_wrist = self.get_wrist_landmarks(left_hand_landmarks)

            # Add wrist landmarks to the results
            normalized_landmarks['left_wrist'] = left_wrist
            normalized_landmarks['right_wrist'] = right_wrist
            
            # Convert normalized landmarks to pixel coordinates
            img_landmarks = self.de_normalize_body_landmarks(normalized_landmarks)
            # Compute vertical and horizontal body distances
            vd, hd = self.get_body_aspect_ratio(img_landmarks)

            # Avoid division by zero
            if hd == 0:
                hd = 1
                print(f"Adjusting BAR {vd}/{hd}")
            # Calculate body aspect ratio (vertical/horizontal distance)
            body_aspect_ratio = vd/hd

            return normalized_landmarks, body_aspect_ratio
        
        else:
            # No pose detected
            return None, None

    def check_body_landmarks(self, coords):
        """Validate that landmark coordinates are normalized (0-1 range)."""
        for key in coords.keys():  # Iterate through each landmark
            for landmark in coords[key]:  # Check each coordinate
                # Clamp x coordinate to 0-1 range
                if coords[key][0] > 1:
                    coords[key][0] = 0
                # Clamp y coordinate to 0-1 range
                if coords[key][1] > 1:
                    coords[key][1] = 0

    def get_wrist_landmarks(self, hand_landmarks):
        """Extract wrist position (first landmark) from hand detection."""
        if hand_landmarks is not None:  # If hand detected
            for idx, landmark in enumerate(hand_landmarks.landmark):
                if idx == 0:  # Wrist is the first landmark
                    return [landmark.x, landmark.y]  # Return normalized coordinates
                
        return [0, 0]  # Return default if no hand detected

    def get_selected_body_landmarks(self, body_landmarks):
        """Extract 9 key body landmarks from full pose detection (normalized 0-1)."""
        # Initialize dictionary with None values for each landmark
        coords = {'front_face':None, 'left_shoulder':None, 'right_shoulder':None, 'left_hip':None, 
                  'right_hip':None, 'left_knee':None, 'right_knee':None, 'right_ankle':None, 'left_ankle':None}

        if body_landmarks is not None:  # If pose detected
            for idx, landmark in enumerate(body_landmarks.landmark):
                # Extract specific landmarks by their MediaPipe indices
                if idx == 0:
                    coords['front_face'] = [landmark.x, landmark.y]
                elif idx == 11:
                    coords['left_shoulder'] = [landmark.x, landmark.y]
                elif idx == 12:
                    coords['right_shoulder'] = [landmark.x, landmark.y]
                elif idx == 23:
                    coords['left_hip'] = [landmark.x, landmark.y]
                elif idx == 24:
                    coords['right_hip'] = [landmark.x, landmark.y]
                elif idx == 25:
                    coords['left_knee'] = [landmark.x, landmark.y]
                elif idx == 26:
                    coords['right_knee'] = [landmark.x, landmark.y]
                elif idx == 28:
                    coords['right_ankle'] = [landmark.x, landmark.y]
                elif idx == 27:
                    coords['left_ankle'] = [landmark.x, landmark.y]

            return coords
        else: 
            return None  # No pose detected
        
    def de_normalize_body_landmarks(self, coords):
        """Convert normalized landmarks (0-1) to pixel coordinates."""
        # Initialize dictionary for pixel coordinates
        img_coords = {'front_face':None, 'left_shoulder':None, 'right_shoulder':None, 'left_hip':None, 
                      'right_hip':None, 'left_knee':None, 'right_knee':None, 'right_ankle':None, 'left_ankle':None}

        for key in coords.keys():
            # Multiply normalized coordinates by image dimensions and convert to integers
            img_coords[key] = [int(coords[key][0]*self.image_size[0]), int(coords[key][1]*self.image_size[1])]

        return img_coords

    def get_body_aspect_ratio(self, normalized_landmarks):
        """
        Calculate body aspect ratio (vertical/horizontal distance).
        Determines if person is standing (ratio > 1) or prone (ratio < 1).
        """
        # Compute max vertical distance between shoulders and knees
        vertical_distance = max(abs(normalized_landmarks['left_shoulder'][1] - normalized_landmarks['right_knee'][1]), 
                                abs(normalized_landmarks['right_shoulder'][1] - normalized_landmarks['left_knee'][1]))
        
        # Compute max horizontal distance between shoulders and knees
        horizontal_distance = max(abs(normalized_landmarks['left_shoulder'][0] - normalized_landmarks['right_knee'][0]), 
                                  abs(normalized_landmarks['right_shoulder'][0] - normalized_landmarks['left_knee'][0]))

        return vertical_distance, horizontal_distance

    def check_body_aspect_ratio(self, vd, hd, img_landmarks):
        """Visualize body aspect ratio by drawing lines on the frame."""
        image_out = self.frame.copy()  # Create a copy to draw on
        # Calculate shoulder center point
        center_shoulders = [int((img_landmarks['left_shoulder'][0] + img_landmarks['right_shoulder'][0]) //2), 
                            int(img_landmarks['left_shoulder'][1])]
        # Calculate bottom point of vertical line (shoulder depth)
        end_shoulders = [int((img_landmarks['left_shoulder'][0] + img_landmarks['right_shoulder'][0]) //2), 
                         int(img_landmarks['left_shoulder'][1])+vd]
        # Calculate end point of horizontal line (body width)
        end_body = [int(img_landmarks['right_shoulder'][0] + hd),  
                    int((img_landmarks['right_shoulder'][1] + img_landmarks['right_knee'][1]) //2)]
        # Calculate body center point
        center_body = [int(img_landmarks['right_shoulder'][0]), 
                       int((img_landmarks['right_shoulder'][1] + img_landmarks['right_knee'][1]) //2)]

        print(center_shoulders)
        print(center_body)
        # Draw vertical line in red
        cv2.line(image_out, center_shoulders, end_shoulders, (0,0,255), 3)
        # Draw horizontal line in red
        cv2.line(image_out, center_body, end_body, (0,0,255), 3)

        return image_out  # Return image with drawn lines
