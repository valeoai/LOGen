from torch.utils.data import Dataset
from nuscenes.utils.geometry_utils import points_in_box
from nuscenes.utils.data_classes import Box, Quaternion
from . import provider
import torch
import torch.nn as nn
import numpy as np
import glob
import json
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

class_mapping_full = {
    'human.pedestrian.adult':0,
    'human.pedestrian.child':0,
    'human.pedestrian.construction_worker':0,
    'human.pedestrian.personal_mobility':0,
    'human.pedestrian.police_officer':0,
    'human.pedestrian.stroller':0,
    'human.pedestrian.wheelchair':0,
    'movable_object.barrier':1,
    'movable_object.pushable_pullable':2,
    'movable_object.trafficcone':3,
    'vehicle.bicycle':4,
    'vehicle.bus.bendy':5,
    'vehicle.bus.rigid':5,
    'car': 6,
    'vehicle.car':6,
    'vehicle.emergency.police':6,
    'vehicle.construction':7,
    'vehicle.motorcycle':8,
    'vehicle.trailer':9,
    'vehicle.truck':10,
    'vehicle.emergency.ambulance':10,
    'movable_object.debris':11,
    'static_object.bicycle_rack':12,
    'animal':13,
}
class_mapping_cleaned = {
  'human.pedestrian.adult':0,
  'human.pedestrian.child':0,
  'human.pedestrian.construction_worker':0,
  'human.pedestrian.personal_mobility':0,
  'human.pedestrian.police_officer':0,
  'human.pedestrian.stroller':0,
  'human.pedestrian.wheelchair':0,
  'movable_object.barrier':1,
  'movable_object.pushable_pullable':2,
  'movable_object.trafficcone':3,
  'vehicle.bicycle':4,
  'vehicle.bus.bendy':5,
  'vehicle.bus.rigid':5,
  'car': 6,
  'vehicle.car':6,
  'vehicle.emergency.police':6,
  'vehicle.construction':7,
  'vehicle.motorcycle':8,
  'vehicle.trailer':9,
  'vehicle.truck':10,
}
num_classes = max(class_mapping_cleaned.values()) + 1

def scale_intensity(data):
    data_log_transformed = np.log1p(data) 
    max_intensity = np.log1p(255.0)
    return data_log_transformed / max_intensity

def pc_normalize_axiswise(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    longest_axis_range = np.max(pc, axis=0) - np.min(pc, axis=0)
    longest_axis = np.argmax(longest_axis_range).item()

    # Find the min and max values along this longest axis
    min_val = np.min(pc[:, longest_axis])
    max_val = np.max(pc[:, longest_axis])

    # Normalize the point cloud based on the longest axis
    pc = (pc - min_val) / (max_val - min_val)
    return pc

def cylindrical_to_cartesian(coordinate):
    angle, dist, z = np.split(coordinate,3,axis=0)
    x = dist * np.cos(angle)
    y = dist * np.sin(angle)
    return np.concatenate((x, y, z), axis=0)

def angle_add(angle1, angle2):
    sum_angle = angle1 + angle2
    sum_angle = (sum_angle + np.pi) % (2 * np.pi) - np.pi
    return sum_angle

class NuscenesGeneratedObjectsDataLoader(Dataset):
    def __init__(self, root, split, real_or_original, filtered_instance_ids=None, class_name='vehicle.bicycle', input_channels=4):
        super().__init__()
        if filtered_instance_ids != None:
            with open(filtered_instance_ids, 'r') as r:
                filtered_instance_ids = set(json.load(r)[split])
            self.dirs = []
            raw_dirs = glob.glob(f'{root}/{split}/**')
            for currdir in raw_dirs:
                sample_annotation_token = currdir.split('/')[-1]
                if sample_annotation_token in filtered_instance_ids:
                    self.dirs.append(currdir)
        else:
            self.dirs = glob.glob(f'{root}/{split}/**')
        self.object_name = 'generated_0' if real_or_original == 'generated' else 'original_0'
        self.class_name = class_name
        self.input_channels = input_channels

    def __len__(self):
        return len(self.dirs)
    
    def __getitem__(self, index):
        curr_dir = self.dirs[index]
        object_data = np.loadtxt(f'{curr_dir}/{self.object_name}.txt', dtype=np.float32)
        condition = np.loadtxt(f'{curr_dir}/conditions_0.txt', dtype=np.float32)
        yaw = condition[-1]
        object_points = object_data[:,:3]
        intensity = object_data[:, 3]
        intensity = scale_intensity(intensity)
        
        center = condition[0:3]
        center[0] = angle_add(center[0], yaw)
        center = cylindrical_to_cartesian(center)
        object_points -= center
        cos_yaw = np.cos(-yaw)
        sin_yaw = np.sin(-yaw)
        rotation_matrix = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw, cos_yaw, 0],
            [0, 0, 1]
        ])
        object_points = np.dot(object_points, rotation_matrix.T)
        object_points = pc_normalize_axiswise(object_points)

        label = class_mapping_cleaned[self.class_name]

        if self.input_channels == 4:
            object_points = np.column_stack((object_points, intensity))

        num_point = object_points.shape[0]
        return object_points, label, num_point

class NuscenesRealObjectsDataLoader(Dataset):
    def __init__(self, data_dir, split, input_channels):
        super().__init__()
        with open(data_dir, 'r') as f:
            self.data_index = json.load(f)[split]

        self.nr_data = len(self.data_index)

        self.split = split
        self.input_channels = input_channels

    def __len__(self):
        return self.nr_data
    
    def __getitem__(self, index):
        object_json = self.data_index[index]
        
        class_name = object_json['class']
        pp = object_json['lidar_data_filepath'].replace("datasets_local", "datasets_master")
        points = np.fromfile(pp, dtype=np.float32).reshape((-1, 5)) #(x, y, z, intensity, ring index)
        center = np.array(object_json['center'])
        size = np.array(object_json['size'])
        rotation_real = np.array(object_json['rotation_real'])
        rotation_imaginary = np.array(object_json['rotation_imaginary'])

        orientation = Quaternion(real=rotation_real, imaginary=rotation_imaginary)
        box = Box(center=center, size=size, orientation=orientation)
        
        points_from_object = points_in_box(box, points=points[:,:3].T)
        object_points = points[points_from_object][:,:3]
        intensity = points[points_from_object][:, 3]
        intensity = scale_intensity(intensity)
        
        object_points -= center
        
        yaw = orientation.yaw_pitch_roll[0]
        cos_yaw = np.cos(-yaw)
        sin_yaw = np.sin(-yaw)
        rotation_matrix = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw, cos_yaw, 0],
            [0, 0, 1]
        ])
        object_points = np.dot(object_points, rotation_matrix.T)

        class_label = class_mapping_cleaned[class_name]
        
        object_points = pc_normalize_axiswise(object_points)
        if self.input_channels == 4:
            object_points = np.column_stack((object_points, intensity))
        num_point = object_points.shape[0]

        return object_points, class_label, num_point
    
class MaxPadObjectCollator(nn.Module):
    def __init__(self, split):
        self.split = split
        return

    def __call__(self, data):
        # "transpose" the  batch(pt, ptn) to batch(pt), batch(ptn)
        batch = list(zip(*data))

        num_points_tensor = torch.Tensor(batch[2])

        pcd_object = batch[0]
        input_channels = pcd_object[0].shape[1]
        max_points = num_points_tensor.max().int().item()
        points = np.zeros((len(pcd_object), max_points, input_channels))
        padding_mask = np.zeros((len(pcd_object), max_points))
        for i, pc in enumerate(pcd_object):
            points[i, :pc.shape[0], :] = pc
            padding_mask [i, :pc.shape[0]] = 1

        if self.split == 'train':
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])

        pcd_object = torch.from_numpy(points).float()
        padding_mask = torch.from_numpy(padding_mask).float()

        return pcd_object, torch.Tensor(batch[1]), padding_mask
    
class NuscenesObjectsDataModule(LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers, data_to_use, test_split=None, filtered_instance_ids=None, class_name='vehicle.bicycle', input_channels=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_to_use = data_to_use
        self.test_split = test_split
        self.filtered_instance_ids = filtered_instance_ids
        self.class_name = class_name
        self.input_channels = input_channels
    
    def prepare_data(self):
        # Augmentations
        pass

    def setup(self, stage=None):
        # Create datasets
        pass

    def train_dataloader(self):
        collate = MaxPadObjectCollator('train')

        data_set = NuscenesRealObjectsDataLoader(
                data_dir=self.data_dir, 
                split='train',
                input_channels=self.input_channels
            )
        loader = DataLoader(data_set, batch_size=self.batch_size, shuffle=True,
                            num_workers=self.num_workers, collate_fn=collate)
        return loader

    def val_dataloader(self, pre_training=True):
        collate = MaxPadObjectCollator('val')

        data_set = NuscenesRealObjectsDataLoader(
                data_dir=self.data_dir, 
                split='val',
                input_channels=self.input_channels
            )
        loader = DataLoader(data_set, batch_size=self.batch_size, shuffle=False,
                            num_workers=self.num_workers, collate_fn=collate)
        return loader

    def test_dataloader(self):
        collate = MaxPadObjectCollator('val')
        data_set = NuscenesGeneratedObjectsDataLoader(
            root=self.data_dir, 
            split=self.test_split,
            real_or_original='generated' if self.data_to_use == 'generated' else 'original',
            filtered_instance_ids=self.filtered_instance_ids,
            class_name=self.class_name
        )
        loader = DataLoader(data_set, batch_size=self.batch_size, shuffle=False,
                             num_workers=self.num_workers, collate_fn=collate)
        return loader