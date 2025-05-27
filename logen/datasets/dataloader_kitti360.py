import os
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import json
import numpy as np
import open3d as o3d
from logen.modules.three_d_helpers import cartesian_to_cylindrical, angle_difference

def scale_intensity(data):
    data_log_transformed = np.log1p(data)
    max_intensity = np.log1p(255.0)
    return data_log_transformed / max_intensity

class Kitti360ObjectsSet(Dataset):
    def __init__(self, 
                 data_dir,
                 split, 
                 points_per_object=None, 
                 recenter=True, 
                 align_objects=False, 
                 relative_angles=False, 
                 stacking_type='duplicate',
                 class_conditional=False,
                 normalize_points=False,
                 input_channels=3,
                 excluded_ids=None,
                 permutation=[],):
        super().__init__()
        with open(data_dir, 'r') as f:
            self.data_index = json.load(f)[split]

        if isinstance(self.data_index, dict):
            if excluded_ids is not None:
                print(f'Before exclusion: {len(self.data_index)} objects')
                self.data_index = [v for k, v in self.data_index.items() if k not in excluded_ids]
                print(f'After exclusion: {len(self.data_index)} objects')
            else:
                self.data_index = list(self.data_index.values())

        if len(permutation) > 0:
            print(f'Applying permutation with {len(permutation)} samples')
            self.data_index = [self.data_index[i] for i in permutation]
            print(f'Final dataset size: {len(self.data_index)}')

        self.nr_data = len(self.data_index)
        self.points_per_object = points_per_object
        self.recenter = recenter
        self.align_objects = align_objects
        self.relative_angles = relative_angles
        self.stacking_type = stacking_type
        self.class_conditional = class_conditional
        self.normalize_points = normalize_points
        self.input_channels = input_channels

    def __len__(self):
        return self.nr_data

    def __getitem__(self, index):
        obj = self.data_index[index]
        
        object_points = np.loadtxt(obj["pointcloud_path"])
        center = np.array(obj['center'])
        size = np.array(obj['size'])
        yaw = obj['rotation_yaw']
        num_points = obj["num_points"]

        if self.points_per_object and self.stacking_type != 'max':
            if object_points.shape[0] > self.points_per_object:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(object_points)
                pcd = pcd.farthest_point_down_sample(self.points_per_object)
                object_points = torch.tensor(np.array(pcd.points))
                padding_mask = torch.ones((object_points.shape[0]))
            else:
                if self.stacking_type == 'pad':
                    padded = torch.zeros((self.points_per_object, 3))
                    padded[:object_points.shape[0]] = torch.from_numpy(object_points)
                    object_points = padded
                    padding_mask = torch.zeros((self.points_per_object))
                    padding_mask[:num_points] = 1
                elif self.stacking_type == 'duplicate':
                    repeats = int(np.ceil(self.points_per_object / object_points.shape[0]))
                    object_points = torch.from_numpy(object_points).repeat(repeats, 1)
                    object_points = object_points[torch.randperm(object_points.shape[0])][:self.points_per_object]
                    padding_mask = torch.ones((object_points.shape[0]))
        else:
            object_points = torch.from_numpy(object_points)
            padding_mask = torch.zeros((object_points.shape[0]))

        if self.recenter:
            object_points -= torch.from_numpy(center)

        center_cyl = cartesian_to_cylindrical(center[None, :]).squeeze(0)
        phi = center_cyl[0]

        if self.align_objects:
            cos_yaw = np.cos(-yaw)
            sin_yaw = np.sin(-yaw)
            rotation_matrix = np.array([
                [cos_yaw, -sin_yaw, 0],
                [sin_yaw, cos_yaw, 0],
                [0, 0, 1]
            ])
            object_points = torch.from_numpy(np.dot(object_points.numpy(), rotation_matrix.T))

        if self.relative_angles:
            center_cyl[0] = angle_difference(phi, yaw)

        if self.class_conditional:
            class_label = torch.tensor(obj['semantic_class'])
        else:
            class_label = torch.ones((1)).int()

        if self.normalize_points:
            ranges = object_points.max(0).values - object_points.min(0).values
            axis = torch.argmax(ranges).item()
            min_val = object_points[:, axis].min()
            max_val = object_points[:, axis].max()
            object_points = (object_points - min_val) / (max_val - min_val)

        if self.input_channels == 4:
            # object_points = torch.cat([object_points, torch.from_numpy(intensity).unsqueeze(1)], dim=1)
            pass

        object_id = obj["pointcloud_path"]

        return [object_points, center_cyl, torch.from_numpy(size), yaw, num_points, class_label, padding_mask, object_id]
