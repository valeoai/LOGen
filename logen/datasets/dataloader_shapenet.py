import os
import torch
import torch.nn.functional
import torch.utils
from torch.utils.data import Dataset
import numpy as np
import glob
import json
from nuscenes.utils.data_classes import Quaternion
from logen.modules.three_d_helpers import cartesian_to_cylindrical

class ShapeNetObjectsSet(Dataset):
    def __init__(self, 
                data_dir, 
                split, 
                points_per_object,
                conditions_data_dir=None,
                relative_angles=False,
                ):
        super().__init__()
        self.data_dir = f'{data_dir}/{split}'
        self.files = glob.glob(f'{self.data_dir}/**.npy')
        self.nr_data = len(self.files)
        self.points_per_object = points_per_object
        if conditions_data_dir != None:
            with open(conditions_data_dir, 'r') as f:
                self.conditions_index = json.load(f)[split]
        else:
            self.conditions_index = None
        self.relative_angles = relative_angles

    def __len__(self):
        return self.nr_data
    
    def get_size(self, shape):
        x = shape[:, 0].max() - shape[:, 0].min()
        y = shape[:, 1].max() - shape[:, 1].min()
        z = shape[:, 2].max() - shape[:, 2].min()
        return np.array((x, y, z))

    def __getitem__(self, index):
        object_points = np.load(self.files[index])

        if self.points_per_object > 0:
            object_points = object_points[torch.randperm(object_points.shape[0])][:self.points_per_object]
        
        num_points = object_points.shape[0]
        rotation_matrix_pitch = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ])
        rotated_point_cloud = object_points.dot(rotation_matrix_pitch.T)
        shape_pcd = rotated_point_cloud
        if self.conditions_index != None:
            random_choice = np.random.randint(len(self.conditions_index))
            condition = self.conditions_index[random_choice]
            center = np.array(condition['center'])
            size = np.array(condition['size'])
            rotation_real = np.array(condition['rotation_real'])
            rotation_imaginary = np.array(condition['rotation_imaginary'])
            orientation = Quaternion(real=rotation_real, imaginary=rotation_imaginary)
            yaw = orientation.yaw_pitch_roll[0]
            cos_yaw = np.cos(-yaw)
            sin_yaw = np.sin(-yaw)
            rotation_matrix = np.array([
                [cos_yaw, -sin_yaw, 0],
                [sin_yaw, cos_yaw, 0],
                [0, 0, 1]
            ])
            rotated_shape = np.dot(shape_pcd, rotation_matrix.T)
            relative_x, relative_y, relative_z = self.get_size(shape_pcd)
            relative_x = size[0] / relative_x
            relative_y = size[1] / relative_y
            relative_z = size[2] / relative_z
            scale = np.array((relative_x, relative_y, relative_z))
            shape_pcd = (rotated_shape * scale[None, :])
            size = self.get_size(shape_pcd)
            center = cartesian_to_cylindrical(center[None,:])[0]
            if self.relative_angles:
                center[0] -= yaw
            orientation = np.zeros(1)
        else:
            size = np.zeros(3) 
            center = np.zeros(3)
            orientation = np.zeros(1)

        padding_mask = torch.zeros((shape_pcd.shape[0]))
        class_name = 'vehicle.motorcycle'
        
        shape_pcd = np.column_stack((shape_pcd, np.zeros((shape_pcd.shape[0], 1))))


        return [shape_pcd, center, torch.from_numpy(size), orientation, num_points, class_name, padding_mask, self.files[index]]