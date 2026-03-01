# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# Import NumPy library for numerical operations
import numpy as np
# Import Skeleton class from the data module for skeleton structure management
from data.skeleton import Skeleton

# Define the MocapDataset class to handle motion capture dataset operations
class MocapDataset:
    # Initialize the dataset with frames per second and skeleton structure
    def __init__(self, fps, skeleton):
        # Store the skeleton object representing the body structure
        self._skeleton = skeleton
        # Store the frames per second (FPS) for the motion capture data
        self._fps = fps
        # Initialize data dictionary to store motion capture data (filled by subclass)
        self._data = None # Must be filled by subclass
        # Initialize cameras dictionary to store camera parameters (filled by subclass)
        self._cameras = None # Must be filled by subclass
    
    # Remove specified joints from the dataset and update skeleton accordingly
    def remove_joints(self, joints_to_remove):
        # Get the indices of joints to keep after removing specified joints
        kept_joints = self._skeleton.remove_joints(joints_to_remove)
        # Iterate through all subjects in the dataset
        for subject in self._data.keys():
            # Iterate through all actions for each subject
            for action in self._data[subject].keys():
                # Get the action data for the current subject
                s = self._data[subject][action]
                # Check if position data exists in the action
                if 'positions' in s:
                    # Filter positions to keep only the specified joints
                    s['positions'] = s['positions'][:, kept_joints]
                
        
    # Retrieve data using dictionary-style indexing
    def __getitem__(self, key):
        # Return the data associated with the given key
        return self._data[key]
        
    # Get all subject identifiers from the dataset
    def subjects(self):
        # Return the keys (subject IDs) from the data dictionary
        return self._data.keys()
    
    # Get the frames per second of the motion capture data
    def fps(self):
        # Return the stored FPS value
        return self._fps
    
    # Get the skeleton structure of the dataset
    def skeleton(self):
        # Return the skeleton object
        return self._skeleton
        
    # Get the camera parameters for the dataset
    def cameras(self):
        # Return the cameras dictionary
        return self._cameras
    
    # Check if the dataset supports semi-supervised learning
    def supports_semi_supervised(self):
        # This method can be overridden
        return False
