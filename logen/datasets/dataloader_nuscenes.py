import os
import torch
import torch.nn.functional
import torch.utils
from torch.utils.data import Dataset
import json
import numpy as np
from nuscenes.utils.geometry_utils import points_in_box
from nuscenes.utils.data_classes import Box, Quaternion
import open3d as o3d
from logen.modules.three_d_helpers import cartesian_to_cylindrical, angle_difference
from logen.modules.class_mapping import class_mapping

def scale_intensity(data):
    data_log_transformed = np.log1p(data) 
    max_intensity = np.log1p(255.0)
    return data_log_transformed / max_intensity
    
class NuscenesObjectsSet(Dataset):
    def __init__(self, 
                data_dir,
                split, 
                volume_expansion=1., 
                input_channels=3,
                excluded_tokens=None,
                permutation=[],
                ):
        super().__init__()
        with open(data_dir, 'r') as f:
            self.data_index = json.load(f)[split]
        
        if isinstance(self.data_index, dict):
            if excluded_tokens != None:
                print(f'Before existing object filtering: {len(self.data_index)} objects')
                self.data_index = [value for key, value in self.data_index.items() if key not in excluded_tokens]
                print(f'After existing object filtering: {len(self.data_index)} objects')
            else:
                self.data_index = list(self.data_index.values())
    
        if len(permutation) > 0:
            print(f'Limiting dataset to {len(permutation)} samples')
            self.data_index = [self.data_index[i] for i in permutation]
            print(f'After limiting, length of dataset is {len(self.data_index)}')

        self.nr_data = len(self.data_index)
        self.volume_expansion = volume_expansion
        self.input_channels = input_channels

    def __len__(self):
        return self.nr_data
    
    def __getitem__(self, index):
        object_json = self.data_index[index]
        
        class_name = object_json['class']
        points = np.fromfile(object_json['lidar_data_filepath'], dtype=np.float32).reshape((-1, 5)) #(x, y, z, intensity, ring index)
        mask = np.load(object_json['object_sample_index'])
        center = np.array(object_json['center'])
        size = np.array(object_json['size'])
        rotation_real = np.array(object_json['rotation_real'])
        rotation_imaginary = np.array(object_json['rotation_imaginary'])
        orientation = Quaternion(real=rotation_real, imaginary=rotation_imaginary)
        
        object_points = points[mask][:,:3]
        intensity = points[mask][:, 3]
        intensity = scale_intensity(intensity)
        
        num_points = object_points.shape[0]

        padding_mask = torch.zeros((object_points.shape[0]))
        
        object_points -= center
        
        center = cartesian_to_cylindrical(center[None,:]).squeeze(0)
        phi = center[0]
        yaw = orientation.yaw_pitch_roll[0]
        
        cos_yaw = np.cos(-yaw)
        sin_yaw = np.sin(-yaw)
        rotation_matrix = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw, cos_yaw, 0],
            [0, 0, 1]
        ])
        object_points = np.dot(object_points, rotation_matrix.T)

        center[0] = angle_difference(phi, yaw)

        class_label = torch.tensor(class_mapping[class_name])

        if self.input_channels == 4:
            object_points = np.column_stack((object_points, intensity))

        return [object_points, center, torch.from_numpy(size), yaw, num_points, class_label, padding_mask, object_json['instance_token']]