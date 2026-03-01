###############################################################################
# Pose Transformers (POTR): Human Motion Prediction with Non-Autoregressive 
# Transformers
# 
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by 
# Angel Martinez <angel.martinez@idiap.ch>,
# 
# This file is part of 
# POTR: Human Motion Prediction with Non-Autoregressive Transformers
# 
# POTR is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
# 
# POTR is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with POTR. If not, see <http://www.gnu.org/licenses/>.
###############################################################################

"""Converting txt files of H36M dataset to numpy matrices."""

# Import numpy for numerical operations and array handling
import numpy as np
# Import os for file system operations and path handling
import os
# Import argparse for command-line argument parsing
import argparse


if __name__ == '__main__':
  # Create argument parser for command-line arguments
  parser =  argparse.ArgumentParser()
  # Add data_path argument for source data directory
  parser.add_argument('--data_path', type=str, default=None)
  # Add output_path argument for destination directory
  parser.add_argument('--output_path', type=str, default=None)
  # Parse command-line arguments
  args = parser.parse_args()

  # Create list of all subdirectories in the data_path
  data_dirs = [os.path.join(args.data_path, x) 
                for x in os.listdir(args.data_path)]

  # Iterate through each directory in data_dirs
  for d in data_dirs:
    # Create list of all files in current directory
    dirs = [os.path.join(d, x) for x in os.listdir(d)]
    # Extract last 3 path components for directory structure
    dir_path = d.split('/')[-3:]

    # Construct output directory path maintaining input structure
    out_path = os.path.join(args.output_path, '/'.join(dir_path))
    # Create output directory if it doesn't exist
    os.makedirs(out_path, exist_ok=True)
    # Iterate through each file in current directory
    for dd in dirs:
      # Print status message with file path being loaded
      print('[ANMG/D] Loading file:', dd)
      # Load txt file as numpy array and convert to float32 type
      txt_mat = np.loadtxt(dd, delimiter=',').astype(np.float32)
      # Extract filename without extension
      in_file = dd.split('/')[-1].split('.')[0]
      # Construct output filepath with .npy extension
      out_file = os.path.join(out_path, in_file+'.npy')
      # Print status message with output file path
      print('[ANMG/D] Saving to:', out_file)
      # Save numpy array to .npy file
      np.save(out_file, txt_mat)
