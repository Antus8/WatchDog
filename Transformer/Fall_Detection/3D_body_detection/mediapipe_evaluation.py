# Import OpenCV for video capture and image processing
import cv2
# Import time module for measuring performance
import time
# Import NumPy for numerical operations
import numpy as np
# Import MediaPipe for pose and holistic detection
import mediapipe as mp
# Import Matplotlib for plotting and visualization
import matplotlib.pyplot as plt
# Import 3D plotting tools from Matplotlib
from mpl_toolkits.mplot3d import Axes3D


class PoseEstimator3D:
    """Class for estimating 3D body pose using MediaPipe models."""
    
    def __init__(self):
        """Initialize the PoseEstimator3D with MediaPipe models and tracking lists."""
        # Store image dimensions
        self.image_size = None

        # Initialize MediaPipe Pose model
        self.mp_pose = mp.solutions.pose
        # Create Pose detector with 85% confidence threshold
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.85)

        # Initialize MediaPipe Holistic model
        self.mp_holistic = mp.solutions.holistic
        # Create Holistic detector with 85% confidence threshold
        self.holistic = self.mp_holistic.Holistic(min_detection_confidence=0.85)

        # Initialize MediaPipe drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils

        # Lists to store inference times for performance comparison
        self.mp_pose_times, self.mp_holistic_times = [], []


    def process_webcam(self):
        """Capture video from webcam and process frames with pose estimation models."""
        # Create figure for 3D pose visualization
        pose_fig = plt.figure()
        # Add 3D subplot for pose landmarks
        pose_ax = pose_fig.add_subplot(111, projection='3d')

        # Create figure for 3D holistic visualization
        holistic_fig = plt.figure()
        # Add 3D subplot for holistic landmarks
        holistic_ax = holistic_fig.add_subplot(111, projection='3d')

        # Initialize video capture from default webcam (0)
        cap = cv2.VideoCapture(0)
        # Loop while camera is open
        while cap.isOpened():
            # Read frame from camera
            ret, frame = cap.read()
            # Store frame dimensions [width, height]
            self.image_size = [frame.shape[1], frame.shape[0]]
            # Check if frame was read successfully
            if not ret:
                print("Error in camera reading...")
                break

            # Convert frame from BGR to RGB color space
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # COMPUTE BODY LANDMARKS USING POSE MODEL
            # Record start time for pose inference
            pose_init = time.time()
            # Process frame with Pose model
            pose_results = self.pose.process(frame)
            # Extract selected body landmarks from pose results
            normalized_pose_landmarks = self.get_selected_pose_landmarks(pose_results.pose_landmarks)
            # Store inference time
            self.mp_pose_times.append(time.time() - pose_init)

            # COMPUTE BODY LANDMARKS USING HOLISTIC MODEL
            # Record start time for holistic inference
            holistic_init = time.time()
            # Process frame with Holistic model
            holistic_results = self.holistic.process(frame)
            # Extract body pose landmarks from holistic results
            body_pose_landmarks = holistic_results.pose_landmarks
            # Get selected body pose landmarks
            normalized_holistic_landmarks = self.get_selected_pose_landmarks(body_pose_landmarks)

            # Extract right hand landmarks
            right_hand_landmarks = holistic_results.right_hand_landmarks
            # Extract left hand landmarks
            left_hand_landmarks = holistic_results.left_hand_landmarks
            # Add right wrist landmark to holistic landmarks
            normalized_holistic_landmarks['right_wrist'] = self.get_wrist_landmarks(right_hand_landmarks)
            # Add left wrist landmark to holistic landmarks
            normalized_holistic_landmarks['left_wrist'] = self.get_wrist_landmarks(left_hand_landmarks)
            # Store inference time
            self.mp_holistic_times.append(time.time() - holistic_init)

            # [COMMENTED OUT] Check if coordinates match between models
            # if self.check_coords(normalized_pose_landmarks, normalized_holistic_landmarks):
            #     print(self.check_coords(normalized_pose_landmarks, normalized_holistic_landmarks))

            # [COMMENTED OUT] Draw landmarks on frames
            # pose_image_out = self.draw_selected_landmarks(self.de_normalize_body_landmarks(normalized_pose_landmarks) , frame.copy())
            # holistic_image_out = self.draw_selected_landmarks(self.de_normalize_body_landmarks(normalized_holistic_landmarks) , frame.copy())
            # cv2.imshow('Pose Estimation', cv2.cvtColor(pose_image_out, cv2.COLOR_RGB2BGR))
            # cv2.imshow('Holistic Estimation',  cv2.cvtColor(holistic_image_out, cv2.COLOR_RGB2BGR))

            # [COMMENTED OUT] 3D PLOTS
            # pose_ax = self.plot_3D_landmarks(pose_ax, normalized_pose_landmarks)
            # holistic_ax = self.plot_3D_landmarks(holistic_ax, normalized_holistic_landmarks)
            # plt.pause(0.01)
            # plt.show(block=False)
            
            # Display current frame
            cv2.imshow('Frame',  cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            # Break loop if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release video capture resource
        cap.release()
        # Close all OpenCV windows
        cv2.destroyAllWindows()

        # Print collected inference times
        print(self.mp_pose_times)
        print(self.mp_holistic_times)
        # Plot performance comparison
        self.plot_time_comparison()


    def plot_time_comparison(self):
        """Plot inference time comparison between Pose and Holistic models."""
        # Create new figure for comparison plot
        comparison_fig = plt.figure()
        # Add subplot for the comparison
        comparison_ax = comparison_fig.add_subplot(111)

        # Calculate average inference time for Pose model
        pose_avg = np.mean(self.mp_pose_times)
        # Calculate average inference time for Holistic model
        holistic_avg = np.mean(self.mp_holistic_times)

        # Create array of frame indices starting from 1
        time_index = np.arange(1, len(self.mp_pose_times) + 1)

        # Plot inference times for Pose model with circle markers
        comparison_ax.plot(time_index, self.mp_pose_times, label='MP_POSE Times', marker='o')
        # Plot inference times for Holistic model with X markers
        comparison_ax.plot(time_index, self.mp_holistic_times, label='MP_HOLISTIC Times', marker='x')

        # Plot average line for Pose model
        comparison_ax.axhline(y=pose_avg, color='blue', linestyle='--', label=f'MP_POSE Avg: {pose_avg:.2f}')
        # Plot average line for Holistic model
        comparison_ax.axhline(y=holistic_avg, color='orange', linestyle='--', label=f'MP_HOLISTIC Avg: {holistic_avg:.2f}')

        # Set X-axis label
        comparison_ax.set_xlabel('Time')
        # Set Y-axis label
        comparison_ax.set_ylabel('Landmarks detection time (per frame)')
        # Set plot title
        plt.title('MediaPipe Landmark detection time comparison')

        # Add legend to plot
        comparison_ax.legend()
        # [COMMENTED OUT] Save plot as SVG file
        # plt.savefig('comparison_plot.svg', format='svg')
        # Display the plot
        plt.show()


    def plot_3D_landmarks(self, ax, landmarks):
        """Plot 3D body landmarks on given axes.
        
        Args:
            ax: Matplotlib 3D axes object
            landmarks: Dictionary of landmark coordinates
            
        Returns:
            Axes object with plotted landmarks
        """
        # Clear previous plot data
        ax.clear()

        # Iterate through all landmarks
        for key in landmarks.keys():
            # Extract X coordinate
            x = landmarks[key][0]
            # Extract Y coordinate (Z in 3D)
            y = landmarks[key][2]
            # Extract Z coordinate (inverted Y for visualization)
            z = -landmarks[key][1]

            # Plot landmark as 3D point
            ax.scatter(x, y, z)

            # Set X-axis limits
            ax.set_xlim(-1, 1)
            # Set Y-axis limits
            ax.set_ylim(-1, 1)
            # Set Z-axis limits
            ax.set_zlim(-1, 1)
            # Label X-axis
            ax.set_xlabel('X-axis')
            # Label Y-axis
            ax.set_ylabel('Y-axis')
            # Label Z-axis
            ax.set_zlabel('Z-axis')
        
        # Return updated axes object
        return ax


    def check_coords(self, pose_coords, holistic_coords):
        """Compare coordinates from Pose and Holistic models.
        
        Args:
            pose_coords: Landmarks from Pose model
            holistic_coords: Landmarks from Holistic model
            
        Returns:
            Boolean indicating if coordinates match
        """
        # Iterate through all landmark keys
        for key in pose_coords.keys():
            # Check if X or Y coordinates differ between models
            if (pose_coords[key][0] != holistic_coords[key][0]) or (pose_coords[key][1] != holistic_coords[key][1]):
                # Print mismatch message
                print(f"Coords for {key} are different!")
                # Print Pose model coordinates
                print(pose_coords[key])
                # Print Holistic model coordinates
                print(holistic_coords[key])
                # Return False if mismatch found
                return False

        # Return True if all coordinates match
        return True


    def get_selected_pose_landmarks(self, body_landmarks):
        """Extract selected body landmarks from MediaPipe results.
        
        Args:
            body_landmarks: MediaPipe body landmarks object
            
        Returns:
            Dictionary of selected landmark coordinates or None
        """
        # Initialize dictionary with all selected landmark keys
        coords = {'front_face':None, 'left_shoulder':None, 'right_shoulder':None, 'left_hip':None, 'right_hip':None, 'left_knee':None, 
                    'right_knee':None, 'right_ankle':None, 'left_ankle':None, 'right_wrist':None, 'left_wrist':None}

        # Check if landmarks were detected
        if body_landmarks is not None:
            # Iterate through all landmarks with their indices
            for idx, landmark in enumerate(body_landmarks.landmark):
                # Index 0: Nose/Front face
                if idx == 0:
                    coords['front_face'] = [landmark.x, landmark.y, landmark.z]
                # Index 11: Left shoulder
                elif idx == 11:
                    coords['left_shoulder'] = [landmark.x, landmark.y, landmark.z]
                # Index 12: Right shoulder
                elif idx == 12:
                    coords['right_shoulder'] = [landmark.x, landmark.y, landmark.z]
                # Index 23: Left hip
                elif idx == 23:
                    coords['left_hip'] = [landmark.x, landmark.y, landmark.z]
                # Index 24: Right hip
                elif idx == 24:
                    coords['right_hip'] = [landmark.x, landmark.y, landmark.z]
                # Index 25: Left knee
                elif idx == 25:
                    coords['left_knee'] = [landmark.x, landmark.y, landmark.z]
                # Index 26: Right knee
                elif idx == 26:
                    coords['right_knee'] = [landmark.x, landmark.y, landmark.z]
                # Index 28: Right ankle
                elif idx == 28:
                    coords['right_ankle'] = [landmark.x, landmark.y, landmark.z]
                # Index 27: Left ankle
                elif idx == 27:
                    coords['left_ankle'] = [landmark.x, landmark.y, landmark.z]
                # Index 16: Right wrist
                elif idx == 16:
                    coords['right_wrist'] = [landmark.x, landmark.y, landmark.z]
                # Index 15: Left wrist
                elif idx == 15:
                    coords['left_wrist'] = [landmark.x, landmark.y, landmark.z]

            # Return landmarks dictionary
            return coords
        else: 
            # Return None if no landmarks detected
            return None


    def get_wrist_landmarks(self, hand_landmarks):
        """Extract wrist landmark from hand landmarks.
        
        Args:
            hand_landmarks: MediaPipe hand landmarks object
            
        Returns:
            List with wrist coordinates or [0, 0, 0] if not detected
        """
        # Check if hand landmarks were detected
        if hand_landmarks is not None:
            # Iterate through hand landmarks
            for idx, landmark in enumerate(hand_landmarks.landmark):
                # Index 0: Wrist point
                if idx == 0:
                    # Return wrist coordinates
                    return[landmark.x, landmark.y, landmark.z]
        
        # Return zero coordinates if not detected
        return [0, 0, 0]
    

    def draw_selected_landmarks(self, landmarks, image):
        """Draw circles at landmark positions on image.
        
        Args:
            landmarks: Dictionary of landmark pixel coordinates
            image: Image to draw on
            
        Returns:
            Image with drawn landmarks
        """
        # Iterate through all landmarks
        for key in landmarks.keys():
            # Draw circle at landmark position with red color
            cv2.circle(image, (landmarks[key][0], landmarks[key][1]), 4, (0, 0, 255), thickness=-1)

        # Return image with drawn landmarks
        return image


    def de_normalize_body_landmarks(self, coords):
        """Convert normalized coordinates to pixel coordinates.
        
        Args:
            coords: Dictionary of normalized landmark coordinates
            
        Returns:
            Dictionary of pixel coordinates
        """
        # Initialize dictionary for pixel coordinates
        img_coords = {'front_face':None, 'left_shoulder':None, 'right_shoulder':None, 'left_hip':None, 'right_hip':None, 'left_knee':None, 'right_knee':None, 'right_ankle':None, 'left_ankle':None, 'right_wrist':None, 'left_wrist':None}

        # Convert each normalized coordinate to pixel coordinates
        for key in coords.keys():
            # Multiply normalized coords by image dimensions and convert to int
            img_coords[key] = [int(coords[key][0]*self.image_size[0]), int(coords[key][1]*self.image_size[1])]

        # Return pixel coordinates dictionary
        return img_coords


def main():
    """Main function to run pose estimation."""
    # Create PoseEstimator3D instance
    body_pose_estimator = PoseEstimator3D()
    # Start processing webcam frames
    body_pose_estimator.process_webcam()


# Run main function if script is executed directly
if __name__ == "__main__":
    main()