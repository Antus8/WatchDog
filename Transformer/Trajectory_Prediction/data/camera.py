# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# Import NumPy for numerical array operations
import numpy as np
# Import PyTorch for tensor operations and GPU acceleration
import torch

# Import utility functions for wrapping operations across batches
from data.utils2 import wrap
# Import quaternion rotation functions
from data.quaternion import qrot, qinverse

def normalize_screen_coordinates(X, w, h): 
    # Verify input is 2D (x, y coordinates)
    assert X.shape[-1] == 2
    
    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    return X/w*2 - [1, h/w]

    
def image_coordinates(X, w, h):
    # Verify input is 2D (x, y coordinates)
    assert X.shape[-1] == 2
    
    # Reverse camera frame normalization
    return (X + [1, h/w])*w/2
    

def world_to_camera(X, R, t):
    # Invert the rotation represented as quaternion
    Rt = wrap(qinverse, R)
    # Apply rotation and translation to transform from world to camera space
    return wrap(qrot, np.tile(Rt, (*X.shape[:-1], 1)), X - t)

    
def camera_to_world(X, R, t):
    # Apply rotation and translation to transform from camera to world space
    return wrap(qrot, np.tile(R, (*X.shape[:-1], 1)), X) + t

    
def project_to_2d(X, camera_params):
    """
    Project 3D points to 2D using the Human3.6M camera projection function.
    This is a differentiable and batched reimplementation of the original MATLAB script.
    
    Arguments:
    X -- 3D points in *camera space* to transform (N, *, 3)
    camera_params -- intrinsic parameteres (N, 2+2+3+2=9)
    """
    # Verify input points are 3D
    assert X.shape[-1] == 3
    # Verify camera parameters are 2D array
    assert len(camera_params.shape) == 2
    # Verify camera parameters have 9 elements (focal length, principal point, distortion coefficients)
    assert camera_params.shape[-1] == 9
    # Verify batch sizes match
    assert X.shape[0] == camera_params.shape[0]
    
    # Expand camera_params dimensions to match X for broadcasting
    while len(camera_params.shape) < len(X.shape):
        camera_params = camera_params.unsqueeze(1)
    
    # Extract focal length (first 2 elements)
    f = camera_params[..., :2]
    # Extract principal point/image center (elements 2-3)
    c = camera_params[..., 2:4]
    # Extract radial distortion coefficients (elements 4-6)
    k = camera_params[..., 4:7]
    # Extract tangential distortion coefficients (elements 7-8)
    p = camera_params[..., 7:]
    
    # Normalize coordinates by depth (perspective division), clamped to [-1, 1]
    XX = torch.clamp(X[..., :2] / X[..., 2:], min=-1, max=1)
    # Calculate squared radius for distortion calculation
    r2 = torch.sum(XX[..., :2]**2, dim=len(XX.shape)-1, keepdim=True)

    # Apply radial distortion model using polynomial coefficients k
    radial = 1 + torch.sum(k * torch.cat((r2, r2**2, r2**3), dim=len(r2.shape)-1), dim=len(r2.shape)-1, keepdim=True)
    # Calculate tangential distortion correction term
    tan = torch.sum(p*XX, dim=len(XX.shape)-1, keepdim=True)

    # Combine radial and tangential distortions
    XXX = XX*(radial + tan) + p*r2
    
    # Apply focal length scaling and add principal point offset to get final 2D image coordinates
    return f*XXX + c

def project_to_2d_linear(X, camera_params):
    """
    Project 3D points to 2D using only linear parameters (focal length and principal point).
    
    Arguments:
    X -- 3D points in *camera space* to transform (N, *, 3)
    camera_params -- intrinsic parameteres (N, 2+2+3+2=9)
    """
    # Verify input points are 3D
    assert X.shape[-1] == 3
    # Verify camera parameters are 2D array
    assert len(camera_params.shape) == 2
    # Verify camera parameters have 9 elements
    assert camera_params.shape[-1] == 9
    # Verify batch sizes match
    assert X.shape[0] == camera_params.shape[0]
    
    # Expand camera_params dimensions to match X for broadcasting
    while len(camera_params.shape) < len(X.shape):
        camera_params = camera_params.unsqueeze(1)
    
    # Extract focal length (first 2 elements)
    f = camera_params[..., :2]
    # Extract principal point/image center (elements 2-3)
    c = camera_params[..., 2:4]
    
    # Normalize coordinates by depth (perspective division), clamped to [-1, 1]
    XX = torch.clamp(X[..., :2] / X[..., 2:], min=-1, max=1)
    
    # Apply focal length scaling and add principal point offset to get final 2D image coordinates (no distortion)
    return f*XX + c
