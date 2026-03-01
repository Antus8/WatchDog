#!/usr/bin/python3.8
# Specify Python 3.8 as the interpreter

import os
# Import operating system module
import cv2
# Import OpenCV for image processing
import glob
# Import glob for file pattern matching
import math
# Import math module for mathematical operations
import time
# Import time module for time-related functions
import rospy
# Import ROS Python client library
import numpy as np
# Import NumPy for numerical operations
import pyrealsense2
# Import Intel RealSense SDK
from cv_bridge import CvBridge
# Import CvBridge to convert between ROS Image messages and OpenCV images
import matplotlib.pyplot as plt
# Import Matplotlib for plotting (currently unused)
from std_msgs.msg import ColorRGBA
# Import ColorRGBA message type for colors
from geometry_msgs.msg import Point
# Import Point message type for 3D coordinates
from visualization_msgs.msg import Marker
# Import Marker message type for RViz visualization
from sensor_msgs.msg import Image, CameraInfo
# Import Image and CameraInfo message types

import sys
# Import sys module
sys.path.append("~/catkin_ws/src/3d_human_detection/")
# Add custom path to Python path
from src.landmark_extractor import LandmarkExtractor
# Import custom LandmarkExtractor class


class SkeletonTracker3D():
# Define SkeletonTracker3D class

    def __init__(self):
    # Constructor method
        super().__init__()
        # Call parent class constructor
        self.br = CvBridge()
        # Initialize CvBridge instance
        self.rgb_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.rgb_callback)
        # Subscribe to RGB camera topic
        self.depth_sub = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.depth_callback)
        # Subscribe to depth camera topic
        self.camera_info_sub = rospy.Subscriber("/camera/depth/camera_info", CameraInfo, self.camera_info_sub_callback)
        # Subscribe to camera info topic

        self.skeleton_image_pub = rospy.Publisher("/camera/skeleton_image", Image, queue_size=1)
        # Create publisher for skeleton image
        self.depth_image_pub = rospy.Publisher("/camera/depth_filtered", Image, queue_size=1)
        # Create publisher for filtered depth image
        self.marker_pub = rospy.Publisher('visualization_marker', Marker, queue_size=10)
        # Create publisher for RViz markers

        self.camera_info = False
        # Flag to track if camera info has been received

        self.landmark_extractor = LandmarkExtractor()
        # Initialize landmark extractor

        self.landmark_keys = ['front_face', 
        # Define landmark key names
                              'left_shoulder', 
                              'right_shoulder', 
                              'left_hip', 
                              'right_hip',
                              'left_knee',
                              'right_knee',
                              'right_ankle',
                              'left_ankle']
        
        self.previous_xs = []
        # List to store previous x coordinates for smoothing


    def camera_info_sub_callback(self, msg):
    # Callback for camera info subscription
        if not self.camera_info:
        # Check if camera info has not been received yet
            self.camera_info = True
            # Set flag to indicate camera info received
            
            cameraInfo = msg
            # Store camera info message
            self._intrinsics = pyrealsense2.intrinsics()
            # Create RealSense intrinsics object
            self._intrinsics.width = cameraInfo.width
            # Set camera width
            self._intrinsics.height = cameraInfo.height
            # Set camera height
            self._intrinsics.ppx = cameraInfo.K[2]
            # Set principal point x coordinate
            self._intrinsics.ppy = cameraInfo.K[5]
            # Set principal point y coordinate
            self._intrinsics.fx = cameraInfo.K[0]
            # Set focal length x
            self._intrinsics.fy = cameraInfo.K[4]
            # Set focal length y

            self._intrinsics.model  = pyrealsense2.distortion.none
            # Set distortion model to none
            self._intrinsics.coeffs = [i for i in cameraInfo.D]
            # Set distortion coefficients

            rospy.loginfo("Camera info coefficients received...")
            # Log info message



    def rgb_callback(self, msg):
    # Callback for RGB image subscription
        frame = self.br.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        # Convert ROS Image message to OpenCV BGR image

        img_landmarks, body_aspect_ratio = self.landmark_extractor.get_body_landmarks(frame)
        # Extract body landmarks from frame
        # l_hip, r_hip = self.landmark_extractor.get_body_hip(frame)
        # (Commented out) Get hip landmarks
        if img_landmarks is not None:
        # Check if landmarks were detected
            image_out = frame.copy()
            # Create copy of frame for output
            cx, cy = self.get_body_center(img_landmarks)
            # Get body center coordinates

            # VISUALIZATION
            # cv2.circle(image_out, (cx, cy), 10, (0, 255, 0), thickness=-1)
            # (Commented) Draw circle at body center
            # image_out = self.landmark_extractor.draw_selected_landmarks(img_landmarks, image_out)
            # (Commented) Draw landmarks on image

            depth_map = self.last_depth.copy()
            # Create copy of latest depth frame
            depth_map_filtered = cv2.GaussianBlur(depth_map, (5, 5), 0)
            # Apply Gaussian blur to smooth depth map

            self.depth_image_pub.publish(self.br.cv2_to_imgmsg(depth_map_filtered))
            # Publish filtered depth image


            # GET PIXELS around the body center
            center_points = self.get_center_square(cx, cy)
            # Get points in square around center

            skeleton_landmarks_3D = []
            # List to store 3D skeleton landmarks
            for x,y in center_points:
            # Loop through each center point
                cv2.circle(image_out, (x, y), 10, (0, 255, 0), thickness=-1)
                # Draw circle at point (for visualization)
                
                if x == depth_map.shape[1]:
                # Check if x is at image boundary
                    x = x-1
                    # Adjust x to stay within bounds
                if y == depth_map.shape[0]:
                # Check if y is at image boundary
                    y = y-1
                    # Adjust y to stay within bounds
                
                depth = depth_map_filtered[y][x]
                # Get depth value at this pixel

                real_x,real_y,real_z = self.project_landmark_to_3D(x,y,depth)
                # Project 2D pixel to 3D real-world coordinates

                rviz_x = real_z
                # RViz x coordinate from camera z
                rviz_y = -real_x
                # RViz y coordinate from negative camera x
                rviz_z = -real_y
                # RViz z coordinate from negative camera y

                skeleton_landmarks_3D.append([rviz_x/1000,rviz_y/1000,rviz_z/1000])
                # Add 3D landmark in meters to list
            
            x_coords, y_coords, z_coords = zip(*skeleton_landmarks_3D)
            # Unzip 3D landmarks into separate coordinate lists
            avg_x = sum(x_coords) / len(x_coords)
            # Calculate average x coordinate
            avg_y = sum(y_coords) / len(y_coords)
            # Calculate average y coordinate
            # avg_z = sum(z_coords) / len(z_coords)
            # (Commented) Calculate average z coordinate
            # rospy.loginfo(avg_x)
            # (Commented) Log average x

            if len(self.previous_xs) == 3:
            # Check if we have 3 previous x values
                historic_x = np.mean(self.previous_xs)
                # Calculate mean of previous x values

                avg_x = (historic_x + avg_x)/2
                # Average with historical value for smoothing

                # if avg_x <= 0.5:
                # (Commented) Check if x is too small
                #     avg_x = self.previous_xs[2]
                # (Commented) Use previous value

                # if abs(historic_x - avg_x) >= 0.5:# or abs(self.previous_xs[2] - avg_x) <= 0.1:
                # (Commented) Check for large deviation
                #     avg_x = self.previous_xs[2]
                # (Commented) Use previous value
                
                del self.previous_xs[0]
                # Remove oldest value from history

            self.previous_xs.append(avg_x)
            # Add current x value to history

            # # Project only body center
            # self.visualize_points([[skeleton_landmarks_3D[0][0]/100, skeleton_landmarks_3D[0][1]/100, avg_z/100]])
            # (Commented) Visualize single point
            # self.visualize_points([[1, 1, avg_z/10]])
            # (Commented) Visualize test point
            self.visualize_points([[avg_x, avg_y, 0]])
            # Visualize average position point
            

        else:
        # If no landmarks detected
            image_out = frame.copy()
            # Use original frame
        
        self.skeleton_image_pub.publish(self.br.cv2_to_imgmsg(image_out))
        # Publish output image


    def depth_callback(self, msg):
    # Callback for depth image subscription
        depth_frame = self.br.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        # Convert ROS depth message to OpenCV image
        
        self.last_depth = depth_frame
        # Store latest depth frame

        # print(f"DEPTH SHAPE IS {depth_frame.shape}")
        # (Commented) Debug print statement

    
    def project_landmark_to_3D(self, x, y, depth):
    # Project 2D pixel to 3D coordinates using camera intrinsics
        result = pyrealsense2.rs2_deproject_pixel_to_point(self._intrinsics, [x, y], depth)
        # Use RealSense function to deproject pixel to 3D point
        
        # return result[2], -result[0], -result[1] This is for rviz coordinate system
        # (Commented) Alternative coordinate system for RViz
        return result[0], result[1], result[2]
        # Return camera coordinate system (x, y, z)
    


    def get_body_center(self, landmarks):
    # Calculate body center from shoulder and hip landmarks
        ls = landmarks['left_shoulder']
        # Get left shoulder coordinates
        rs = landmarks['right_shoulder']
        # Get right shoulder coordinates
        
        lh = landmarks['left_hip']
        # Get left hip coordinates
        rh = landmarks['right_hip']
        # Get right hip coordinates

        c_x = max(((ls[0]+rs[0])/2), ((lh[0]+rh[0])/2))
        # Center x is max of shoulder and hip center x
        c_y = max(((ls[1]+lh[1])/2), ((rs[1]+rh[1])/2))
        # Center y is max of left and right body center y

        return int(c_x), int(c_y)
        # Return integer coordinates


    def get_center_square(self, x, y):
    # Get points in a square around center coordinates
        square_coords = [(x,y)]
        # Initialize list with center point
        
        for dx in range(-10, 11):
        # Loop from -10 to 10 in x direction
            for dy in range(-10, 11):
            # Loop from -10 to 10 in y direction
                nx, ny = x + dx, y + dy
                # Calculate new x and y coordinates
                
                square_coords.append((nx, ny))
                # Add point to list
        
        return square_coords
        # Return all points in square


    def visualize_points(self, points):
    # Create and publish RViz markers for given points
        # Create a Marker object
        marker = Marker()
        # Initialize new marker
        marker.header.frame_id = "world"
        # Set frame of reference to world
        marker.header.stamp = rospy.Time.now()
        # Set timestamp to current time
        marker.ns = "points"
        # Set namespace
        marker.id = 0
        # Set marker ID
        marker.type = Marker.POINTS
        # Set marker type to points
        marker.action = Marker.ADD
        # Set action to add marker
        
        # Set the scale of the points (size of the marker points)
        marker.scale.x = 0.1
        # Set point width
        marker.scale.y = 0.1
        # Set point height
        
        # Set the color of the points (RGBA format)
        marker.color = ColorRGBA(1.0, 0.0, 0.0, 1.0)
        # Set color to red with full opacity
        
        marker.points = []
        # Initialize empty points list
        # Add points to the marker
        for point in points:
        # Loop through provided points
            p = Point()
            # Create new Point message
            p.x = point[0]
            # Set X coordinate
            p.y = point[1]
            # Set Y coordinate
            p.z = point[2]
            # Set Z coordinate
            marker.points.append(p)
            # Add point to marker
        
        
        marker.header.stamp = rospy.Time.now()
        # Update the timestamp
        self.marker_pub.publish(marker)
        # Publish point marker

        if points:
        # Check if points list is not empty
            label_marker = Marker()
            # Create marker for text label
            label_marker.header.frame_id = "world"
            # Set frame of reference
            label_marker.header.stamp = rospy.Time.now()
            # Set timestamp
            label_marker.ns = "labels"
            # Set namespace
            label_marker.id = 1
            # Set unique ID
            label_marker.type = Marker.TEXT_VIEW_FACING
            # Set type to text facing camera
            label_marker.action = Marker.ADD
            # Set action to add
            
            # Position the label just above the first point
            label_marker.pose.position.x = points[0][0]
            # Set label x position
            label_marker.pose.position.y = points[0][1]
            # Set label y position
            label_marker.pose.position.z = points[0][2] + 0.1
            # Set label z position above point
            
            label_marker.scale.z = 0.1
            # Set text font size
            label_marker.color = ColorRGBA(1.0, 1.0, 1.0, 1.0)
            # Set text color to white
            
            label_marker.text = "human"
            # Set label text
            self.marker_pub.publish(label_marker)
            # Publish label marker

        
            # Publish a green marker at the world origin (0,0,0)
            origin_marker = Marker()
            # Create marker for world origin
            origin_marker.header.frame_id = "world"
            # Set frame of reference
            origin_marker.header.stamp = rospy.Time.now()
            # Set timestamp
            origin_marker.ns = "origin"
            # Set namespace
            origin_marker.id = 2
            # Set unique ID
            origin_marker.type = Marker.SPHERE
            # Set type to sphere for origin marker
            origin_marker.action = Marker.ADD
            # Set action to add

            # Position the marker at the origin
            origin_marker.pose.position.x = 0
            # Set origin x to 0
            origin_marker.pose.position.y = 0
            # Set origin y to 0
            origin_marker.pose.position.z = 0
            # Set origin z to 0

            origin_marker.scale.x = 0.1
            # Set sphere radius
            origin_marker.scale.y = 0.1
            # Set sphere size (uniform)
            origin_marker.scale.z = 0.1
            # Set sphere size (uniform)

            origin_marker.color = ColorRGBA(0.0, 1.0, 0.0, 1.0)
            # Set color to green
            self.marker_pub.publish(origin_marker)
            # Publish origin marker
            


def main():
# Main function entry point
    rospy.init_node("skeleton_tracker_node")
    # Initialize ROS node
    my_node = SkeletonTracker3D()
    # Create SkeletonTracker3D instance
    while not rospy.is_shutdown():
    # Loop while ROS is running
        rospy.spin()
        # Keep node running


if __name__ == "__main__":
# Check if script is run directly
    main()
    # Call main function