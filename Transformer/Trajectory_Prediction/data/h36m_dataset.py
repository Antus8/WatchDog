# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# Import NumPy for numerical operations and array handling
import numpy as np
# Import copy module for deep copying data structures
import copy
# Import Skeleton class from local data module for skeleton structure definition
from data.skeleton import Skeleton
# Import MocapDataset base class for motion capture dataset functionality
from data.mocap_dataset import MocapDataset
# Import camera utility functions for coordinate normalization and transformation
from data.camera import normalize_screen_coordinates, image_coordinates

# Define the Human3.6M skeleton structure with parent-child joint relationships
# parents: list indicating the parent joint index for each joint (-1 for root)
# joints_left: indices of joints on the left side of the body
# joints_right: indices of joints on the right side of the body
h36m_skeleton = Skeleton(parents=[-1,  0,  1,  2,  3,  4,  0,  6,  7,  8,  9,  0, 11, 12, 13, 14, 12,
    16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27, 28, 27, 30],
    joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
    joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31])

# List of camera intrinsic calibration parameters for 4 cameras in Human3.6M dataset
# Each camera has: id, principal point (center), focal length, and distortion coefficients
h36m_cameras_intrinsic_params = [
    {
     'id': '54138969',  # Camera 1 identifier
     'center': [512.54150390625, 515.4514770507812],  # Principal point (cx, cy)
     'focal_length': [1145.0494384765625, 1143.7811279296875],  # Focal length (fx, fy)
     'radial_distortion': [-0.20709891617298126, 0.24777518212795258, -0.0030751503072679043],  # Radial distortion coefficients
     'tangential_distortion': [-0.0009756988729350269, -0.00142447161488235],  # Tangential distortion coefficients
     'res_w': 1000,  # Image width resolution
     'res_h': 1002,  # Image height resolution
     'azimuth': 70, # Only used for visualization
    },
    {
     'id': '55011271',  # Camera 2 identifier
     'center': [508.8486328125, 508.0649108886719],  # Principal point (cx, cy)
     'focal_length': [1149.6756591796875, 1147.5916748046875],  # Focal length (fx, fy)
     'radial_distortion': [-0.1942136287689209, 0.2404085397720337, 0.006819975562393665],  # Radial distortion coefficients
     'tangential_distortion': [-0.0016190266469493508, -0.0027408944442868233],  # Tangential distortion coefficients
     'res_w': 1000,  # Image width resolution
     'res_h': 1000,  # Image height resolution
     'azimuth': -70, # Only used for visualization
    },
    {
     'id': '58860488',  # Camera 3 identifier
     'center': [519.8158569335938, 501.40264892578125],  # Principal point (cx, cy)
     'focal_length': [1149.1407470703125, 1148.7989501953125],  # Focal length (fx, fy)
     'radial_distortion': [-0.2083381861448288, 0.25548800826072693, -0.0024604974314570427],  # Radial distortion coefficients
     'tangential_distortion': [0.0014843869721516967, -0.0007599993259645998],  # Tangential distortion coefficients
     'res_w': 1000,  # Image width resolution
     'res_h': 1000,  # Image height resolution
     'azimuth': 110, # Only used for visualization
    },
    {
     'id': '60457274',  # Camera 4 identifier
     'center': [514.9682006835938, 501.88201904296875],  # Principal point (cx, cy)
     'focal_length': [1145.5113525390625, 1144.77392578125],  # Focal length (fx, fy)
     'radial_distortion': [-0.198384091258049, 0.21832367777824402, -0.008947807364165783],  # Radial distortion coefficients
     'tangential_distortion': [-0.0005872055771760643, -0.0018133620033040643],  # Tangential distortion coefficients
     'res_w': 1000,  # Image width resolution
     'res_h': 1002,  # Image height resolution
     'azimuth': -110, # Only used for visualization
    },
]

# Dictionary containing extrinsic camera parameters (rotation and translation) for multiple subjects
# Each subject has 4 cameras with orientation (quaternion) and translation vectors
h36m_cameras_extrinsic_params = {
    'S1': [
     {
         'orientation': [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088],  # Quaternion (w, x, y, z)
         'translation': [1841.1070556640625, 4955.28466796875, 1563.4454345703125],  # Camera position in world coordinates (mm)
     },
     {
         'orientation': [0.6157187819480896, -0.764836311340332, -0.14833825826644897, 0.11794740706682205],  # Quaternion (w, x, y, z)
         'translation': [1761.278564453125, -5078.0068359375, 1606.2650146484375],  # Camera position in world coordinates (mm)
     },
     {
         'orientation': [0.14651472866535187, -0.14647851884365082, 0.7653023600578308, -0.6094175577163696],  # Quaternion (w, x, y, z)
         'translation': [-1846.7777099609375, 5215.04638671875, 1491.972412109375],  # Camera position in world coordinates (mm)
     },
     {
         'orientation': [0.5834008455276489, -0.7853162288665771, 0.14548823237419128, -0.14749594032764435],  # Quaternion (w, x, y, z)
         'translation': [-1794.7896728515625, -3722.698974609375, 1574.8927001953125],  # Camera position in world coordinates (mm)
     },
    ],
    'S2': [
     {},  # Empty camera parameters for subject 2, camera 1
     {},  # Empty camera parameters for subject 2, camera 2
     {},  # Empty camera parameters for subject 2, camera 3
     {},  # Empty camera parameters for subject 2, camera 4
    ],
    'S3': [
     {},  # Empty camera parameters for subject 3, camera 1
     {},  # Empty camera parameters for subject 3, camera 2
     {},  # Empty camera parameters for subject 3, camera 3
     {},  # Empty camera parameters for subject 3, camera 4
    ],
    'S4': [
     {},  # Empty camera parameters for subject 4, camera 1
     {},  # Empty camera parameters for subject 4, camera 2
     {},  # Empty camera parameters for subject 4, camera 3
     {},  # Empty camera parameters for subject 4, camera 4
    ],
    'S5': [
     {
         'orientation': [0.1467377245426178, -0.162370964884758, -0.7551892995834351, 0.6178938746452332],  # Quaternion (w, x, y, z)
         'translation': [2097.3916015625, 4880.94482421875, 1605.732421875],  # Camera position in world coordinates (mm)
     },
     {
         'orientation': [0.6159758567810059, -0.7626792192459106, -0.15728192031383514, 0.1189815029501915],  # Quaternion (w, x, y, z)
         'translation': [2031.7008056640625, -5167.93310546875, 1612.923095703125],  # Camera position in world coordinates (mm)
     },
     {
         'orientation': [0.14291371405124664, -0.12907841801643372, 0.7678384780883789, -0.6110143065452576],  # Quaternion (w, x, y, z)
         'translation': [-1620.5948486328125, 5171.65869140625, 1496.43701171875],  # Camera position in world coordinates (mm)
     },
     {
         'orientation': [0.5920479893684387, -0.7814217805862427, 0.1274748593568802, -0.15036417543888092],  # Quaternion (w, x, y, z)
         'translation': [-1637.1737060546875, -3867.3173828125, 1547.033203125],  # Camera position in world coordinates (mm)
     },
    ],
    'S6': [
     {
         'orientation': [0.1337897777557373, -0.15692396461963654, -0.7571090459823608, 0.6198879480361938],  # Quaternion (w, x, y, z)
         'translation': [1935.4517822265625, 4950.24560546875, 1618.0838623046875],  # Camera position in world coordinates (mm)
     },
     {
         'orientation': [0.6147197484970093, -0.7628812789916992, -0.16174767911434174, 0.11819244921207428],  # Quaternion (w, x, y, z)
         'translation': [1969.803955078125, -5128.73876953125, 1632.77880859375],  # Camera position in world coordinates (mm)
     },
     {
         'orientation': [0.1529948115348816, -0.13529130816459656, 0.7646096348762512, -0.6112781167030334],  # Quaternion (w, x, y, z)
         'translation': [-1769.596435546875, 5185.361328125, 1476.993408203125],  # Camera position in world coordinates (mm)
     },
     {
         'orientation': [0.5916101336479187, -0.7804774045944214, 0.12832270562648773, -0.1561593860387802],  # Quaternion (w, x, y, z)
         'translation': [-1721.668701171875, -3884.13134765625, 1540.4879150390625],  # Camera position in world coordinates (mm)
     },
    ],
    'S7': [
     {
         'orientation': [0.1435241848230362, -0.1631336808204651, -0.7548328638076782, 0.6188824772834778],  # Quaternion (w, x, y, z)
         'translation': [1974.512939453125, 4926.3544921875, 1597.8326416015625],  # Camera position in world coordinates (mm)
     },
     {
         'orientation': [0.6141672730445862, -0.7638262510299683, -0.1596645563840866, 0.1177929937839508],  # Quaternion (w, x, y, z)
         'translation': [1937.0584716796875, -5119.7900390625, 1631.5665283203125],  # Camera position in world coordinates (mm)
     },
     {
         'orientation': [0.14550060033798218, -0.12874816358089447, 0.7660516500473022, -0.6127139329910278],  # Quaternion (w, x, y, z)
         'translation': [-1741.8111572265625, 5208.24951171875, 1464.8245849609375],  # Camera position in world coordinates (mm)
     },
     {
         'orientation': [0.5912848114967346, -0.7821764349937439, 0.12445473670959473, -0.15196487307548523],  # Quaternion (w, x, y, z)
         'translation': [-1734.7105712890625, -3832.42138671875, 1548.5830078125],  # Camera position in world coordinates (mm)
     },
    ],
    'S8': [
     {
         'orientation': [0.14110587537288666, -0.15589867532253265, -0.7561917304992676, 0.619644045829773],  # Quaternion (w, x, y, z)
         'translation': [2150.65185546875, 4896.1611328125, 1611.9046630859375],  # Camera position in world coordinates (mm)
     },
     {
         'orientation': [0.6169601678848267, -0.7647668123245239, -0.14846350252628326, 0.11158157885074615],  # Quaternion (w, x, y, z)
         'translation': [2219.965576171875, -5148.453125, 1613.0440673828125],  # Camera position in world coordinates (mm)
     },
     {
         'orientation': [0.1471444070339203, -0.13377119600772858, 0.7670128345489502, -0.6100369691848755],  # Quaternion (w, x, y, z)
         'translation': [-1571.2215576171875, 5137.0185546875, 1498.1761474609375],  # Camera position in world coordinates (mm)
     },
     {
         'orientation': [0.5927824378013611, -0.7825870513916016, 0.12147816270589828, -0.14631995558738708],  # Quaternion (w, x, y, z)
         'translation': [-1476.913330078125, -3896.7412109375, 1547.97216796875],  # Camera position in world coordinates (mm)
     },
    ],
    'S9': [
     {
         'orientation': [0.15540587902069092, -0.15548215806484222, -0.7532095313072205, 0.6199594736099243],  # Quaternion (w, x, y, z)
         'translation': [2044.45849609375, 4935.1171875, 1481.2275390625],  # Camera position in world coordinates (mm)
     },
     {
         'orientation': [0.618784487247467, -0.7634735107421875, -0.14132238924503326, 0.11933968216180801],  # Quaternion (w, x, y, z)
         'translation': [1990.959716796875, -5123.810546875, 1568.8048095703125],  # Camera position in world coordinates (mm)
     },
     {
         'orientation': [0.13357827067375183, -0.1367100477218628, 0.7689454555511475, -0.6100738644599915],  # Quaternion (w, x, y, z)
         'translation': [-1670.9921875, 5211.98583984375, 1528.387939453125],  # Camera position in world coordinates (mm)
     },
     {
         'orientation': [0.5879399180412292, -0.7823407053947449, 0.1427614390850067, -0.14794869720935822],  # Quaternion (w, x, y, z)
         'translation': [-1696.04345703125, -3827.099853515625, 1591.4127197265625],  # Camera position in world coordinates (mm)
     },
    ],
    'S11': [
     {
         'orientation': [0.15232472121715546, -0.15442320704460144, -0.7547563314437866, 0.6191070079803467],  # Quaternion (w, x, y, z)
         'translation': [2098.440185546875, 4926.5546875, 1500.278564453125],  # Camera position in world coordinates (mm)
     },
     {
         'orientation': [0.6189449429512024, -0.7600917220115662, -0.15300633013248444, 0.1255258321762085],  # Quaternion (w, x, y, z)
         'translation': [2083.182373046875, -4912.1728515625, 1561.07861328125],  # Camera position in world coordinates (mm)
     },
     {
         'orientation': [0.14943228662014008, -0.15650227665901184, 0.7681233882904053, -0.6026304364204407],  # Quaternion (w, x, y, z)
         'translation': [-1609.8153076171875, 5177.3359375, 1537.896728515625],  # Camera position in world coordinates (mm)
     },
     {
         'orientation': [0.5894251465797424, -0.7818877100944519, 0.13991211354732513, -0.14715361595153809],  # Quaternion (w, x, y, z)
         'translation': [-1590.738037109375, -3854.1689453125, 1578.017578125],  # Camera position in world coordinates (mm)
     },
    ],
}

# Class definition for Human3.6M dataset, inherits from MocapDataset base class
class Human36mDataset(MocapDataset):
    def __init__(self, path, remove_static_joints=True):
     # Initialize parent class with 50 fps frame rate and predefined skeleton
     super().__init__(fps=50, skeleton=h36m_skeleton)
     
     # Create a deep copy of extrinsic camera parameters to avoid modifying original data
     self._cameras = copy.deepcopy(h36m_cameras_extrinsic_params)
     # Iterate through all camera sets for each subject
     for cameras in self._cameras.values():
         # Iterate through each of the 4 cameras
         for i, cam in enumerate(cameras):
          # Merge intrinsic parameters from the corresponding camera index
          cam.update(h36m_cameras_intrinsic_params[i])
          # Convert numerical parameters to float32 numpy arrays, excluding metadata
          for k, v in cam.items():
              if k not in ['id', 'res_w', 'res_h']:
               cam[k] = np.array(v, dtype='float32')
          
          # Normalize camera frame: convert image coordinates to normalized coordinates (-1 to 1)
          cam['center'] = normalize_screen_coordinates(cam['center'], w=cam['res_w'], h=cam['res_h']).astype('float32')
          # Normalize focal length relative to image width
          cam['focal_length'] = cam['focal_length']/cam['res_w']*2
          # Convert translation from millimeters to meters
          if 'translation' in cam:
              cam['translation'] = cam['translation']/1000 # mm to meters
          
          # Concatenate all intrinsic parameters into a single vector for efficient processing
          cam['intrinsic'] = np.concatenate((cam['focal_length'],
                              cam['center'],
                              cam['radial_distortion'],
                              cam['tangential_distortion']))
     
     # Load pre-processed 3D position data from numpy file
     data = np.load(path, allow_pickle=True)['positions_3d'].item()
     
     # Initialize dictionary to store dataset organized by subject and action
     self._data = {}
     # Iterate through each subject in the dataset
     for subject, actions in data.items():
         # Create subject entry in dataset
         self._data[subject] = {}
         # Iterate through each action performed by the subject
         for action_name, positions in actions.items():
          # Store 3D joint positions and corresponding camera parameters for each action
          self._data[subject][action_name] = {
              'positions': positions,  # 3D joint coordinates
              'cameras': self._cameras[subject],  # Camera parameters for this subject
          }
# =============================================================================
#                 
#         if remove_static_joints:
#             # Bring the skeleton to 17 joints instead of the original 32
#             self.remove_joints([4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31])
#             
#             # Rewire shoulders to the correct parents
#             self._skeleton._parents[11] = 8
#             self._skeleton._parents[14] = 8
#             
# =============================================================================
    # Method to indicate that this dataset supports semi-supervised learning approaches
    def supports_semi_supervised(self):
     return True
