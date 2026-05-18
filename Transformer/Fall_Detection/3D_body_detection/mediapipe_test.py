# Import OpenCV library for image and video processing
import cv2

# Import MediaPipe library for pose estimation solutions
import mediapipe as mp

# Import Matplotlib for plotting and visualization
import matplotlib.pyplot as plt

# Import 3D plotting toolkit from Matplotlib
from mpl_toolkits.mplot3d import Axes3D


# Define a class for 3D pose estimation using MediaPipe
class PoseEstimator3D:
    """
    A class to perform 3D pose estimation from webcam video feed.
    Detects human body landmarks and visualizes them in 3D space.
    """

    # Initialize the pose estimator with MediaPipe models
    def __init__(self):
        # Load MediaPipe pose solution
        self.mp_pose = mp.solutions.pose
        # Create a Pose object for landmark detection
        self.pose = self.mp_pose.Pose()
        # Load drawing utilities for landmark visualization
        self.mp_drawing = mp.solutions.drawing_utils

    # Main method to process video from webcam
    def process_webcam(self):
        # Create a new figure for 3D plotting
        fig = plt.figure()
        # Add a 3D subplot to the figure
        ax = fig.add_subplot(111, projection='3d')

        # Initialize video capture from default webcam (index 0)
        cap = cv2.VideoCapture(0)
        
        # Loop while webcam is open
        while cap.isOpened():
            # Read a frame from the webcam
            ret, frame = cap.read()
            
            # Break loop if frame is not read successfully
            if not ret:
                break
            
            # Convert frame from BGR to RGB color space
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame to detect pose landmarks
            results = self.pose.process(frame_rgb)
            
            # Draw detected landmarks on the frame
            annotated_frame = self._draw_landmarks(frame, results.pose_landmarks)
            
            # Display the annotated frame in a window
            cv2.imshow('Pose Estimation', annotated_frame)

            # Clear the 3D plot for next frame
            ax.clear()

            # Check if landmarks were detected
            if results.pose_landmarks:
                # Extract all detected landmarks
                landmarks = results.pose_landmarks.landmark
                
                # Extract x coordinates from all landmarks
                x = [landmark.x for landmark in landmarks]
                # Extract z coordinates (mapped to y-axis) from all landmarks
                y = [landmark.z for landmark in landmarks]
                # Extract y coordinates (negated) and map to z-axis for proper orientation
                z = [-landmark.y for landmark in landmarks]

                # Plot the 3D landmarks as scatter points
                ax.scatter(x, y, z)

            # Set the range for x-axis
            ax.set_xlim(-1, 1)
            # Set the range for y-axis
            ax.set_ylim(-1, 1)
            # Set the range for z-axis
            ax.set_zlim(-1, 1)
            
            # Label the x-axis
            ax.set_xlabel('X-axis')
            # Label the y-axis
            ax.set_ylabel('Y-axis')
            # Label the z-axis
            ax.set_zlabel('Z-axis')

            # Pause for 0.01 seconds to update the plot
            plt.pause(0.01)
            # Display the plot without blocking execution
            plt.show(block=False)

            # Check if 'q' key is pressed to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release the webcam resource
        cap.release()
        # Close all OpenCV windows
        cv2.destroyAllWindows()

    # Helper method to draw pose landmarks on an image
    def _draw_landmarks(self, image, landmarks):
        """
        Draws detected pose landmarks and connections on the image.
        
        Args:
            image: The input image frame
            landmarks: Detected pose landmarks from MediaPipe
            
        Returns:
            The image with drawn landmarks, or original image if no landmarks
        """
        
        # Check if landmarks were detected
        if landmarks is not None:
            # Create a copy of the image to avoid modifying the original
            annotated_image = image.copy()
            
            # Draw landmarks and their connections on the image
            self.mp_drawing.draw_landmarks(
                annotated_image,
                landmarks,
                self.mp_pose.POSE_CONNECTIONS)
            
            # Return the image with drawn landmarks
            return annotated_image
        else:
            # Return original image if no landmarks detected
            return image


# Entry point of the script
def main():
    # Create an instance of the PoseEstimator3D class
    body_pose_estimator = PoseEstimator3D()
    # Start processing webcam feed
    body_pose_estimator.process_webcam()


# Run main function only if script is executed directly (not imported)
if __name__ == "__main__":
    main()