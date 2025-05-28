import os
import json
import numpy as np
from nuscenes.utils.geometry_utils import points_in_box
from nuscenes.nuscenes import NuScenes
import nuscenes.utils.splits as splits
from tqdm import tqdm
import sys
import glob

def parse_objects_from_nuscenes(points_threshold, object_name, object_tag, range_to_use):
    # Path to the dataset
    dataroot = '/datasets_local/nuscenes'

    # Initialize the nuScenes class
    nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True)

    # Directory to save extracted data
    output_dir = f'{object_name}_from_nuscenes'
    os.makedirs(output_dir, exist_ok=True)

    train_split = set(splits.train)

    object_lidar_data = {'train':[], 'val':[]}
    for i in tqdm(range_to_use):
        sample = nusc.sample[i]
        scene_token = sample['scene_token']
        sample_token = sample['token']
        sample_data_lidar_token = sample['data']['LIDAR_TOP']
        scene_name = nusc.get('scene', scene_token)['name']
        split = 'train' if scene_name in train_split else 'val'

        objects = nusc.get_sample_data(sample_data_lidar_token)[1]
        lidar_data = nusc.get('sample_data', sample_data_lidar_token)
        lidar_filepath = os.path.join(dataroot, lidar_data['filename'])
        lidarseg_label_filename = os.path.join(nusc.dataroot, nusc.get('lidarseg', sample_data_lidar_token)['filename'])
        points = np.fromfile(lidar_filepath, dtype=np.float32).reshape((-1, 5)) #(x, y, z, intensity, ring index)
        labels = np.fromfile(lidarseg_label_filename, dtype=np.uint8).reshape((-1, 1))
        for object in objects:
            if object_tag in object.name:
                points_from_object = points_in_box(object, points=points[:,:3].T)
                object_points = points[points_from_object][:,:3]
                object_isolated_label_mask = (labels[points_from_object] == class_mapping[object.name]).flatten()
                object_points = object_points[object_isolated_label_mask]
                
                num_lidar_points = len(object_points)
                if num_lidar_points < points_threshold:
                    continue

                object_info = {
                    'instance_token': object.token,
                    'sample_token': sample_token,
                    'scene_token': scene_token,
                    'sample_data_lidar_token': sample_data_lidar_token,
                    'lidar_data_filepath': lidar_filepath,
                    'lidarseg_label_filepath': lidarseg_label_filename,
                    'class': object.name,
                    'center': object.center.tolist(),
                    'size': object.wlh.tolist(),
                    'rotation_real': object.orientation.real.tolist(),
                    'rotation_imaginary': object.orientation.imaginary.tolist(),
                    'num_lidar_points': num_lidar_points,
                }
                object_lidar_data[split].append(object_info)

    print(f"After parsing, {len(object_lidar_data['train'])} objects in train, {len(object_lidar_data['val'])} in val")
    with open(f'{output_dir}/{object_name}_from_nuscenes_train_val.json', 'w') as fp:
        json.dump(object_lidar_data, fp)

    return object_lidar_data, output_dir

def parse_largest_x_from_dataset(output_dir, object_name, object_lidar_data, top_x):
    reduced_train_val_objects = {'train':[], 'val':[]}
    train_objects = object_lidar_data['train']
    val_objects = object_lidar_data['val']
    print("Sorting object lidar data")
    train_sorted_by_num_points = sorted(train_objects, key=lambda x: x['num_lidar_points'], reverse=True)
    val_sorted_by_num_points = sorted(val_objects, key=lambda x: x['num_lidar_points'], reverse=True)
    print("Taking top x from object lidar data")
    reduced_train_val_objects['train'] = train_sorted_by_num_points[:top_x]
    reduced_train_val_objects['val'] = val_sorted_by_num_points[:top_x]

    with open(f'{output_dir}/{object_name}_from_nuscenes_train_val_reduced.json', 'w') as fp:
        json.dump(reduced_train_val_objects, fp)

id_to_class = {0: 'noise',
            1: 'animal',
            2: 'human.pedestrian.adult',
            3: 'human.pedestrian.child',
            4: 'human.pedestrian.construction_worker',
            5: 'human.pedestrian.personal_mobility',
            6: 'human.pedestrian.police_officer',
            7: 'human.pedestrian.stroller',
            8: 'human.pedestrian.wheelchair',
            9: 'movable_object.barrier',
            10: 'movable_object.debris',
            11: 'movable_object.pushable_pullable',
            12: 'movable_object.trafficcone',
            13: 'static_object.bicycle_rack',
            14: 'vehicle.bicycle',
            15: 'vehicle.bus.bendy',
            16: 'vehicle.bus.rigid',
            17: 'vehicle.car',
            18: 'vehicle.construction',
            19: 'vehicle.emergency.ambulance',
            20: 'vehicle.emergency.police',
            21: 'vehicle.motorcycle',
            22: 'vehicle.trailer',
            23: 'vehicle.truck',
            24: 'flat.driveable_surface',
            25: 'flat.other',
            26: 'flat.sidewalk',
            27: 'flat.terrain',
            28: 'static.manmade',
            29: 'static.other',
            30: 'static.vegetation',
            31: 'vehicle.ego'
            }
class_mapping = {v: k for k,v in id_to_class.items()}

if __name__ == '__main__':
    num_jobs = int(sys.argv[1])
    range_index = int(sys.argv[2])
    points_threshold = 5
    object_name = f'all_objects_5pts_class_filtered_range_{range_index}'
    object_tag = ''

    total_scenes = 34149
    average_per_index_scenes = total_scenes // num_jobs
    start_index = range_index * average_per_index_scenes
    end_index = (range_index+1)*average_per_index_scenes
    if range_index == num_jobs - 1:
        end_index = total_scenes
    curr_range = range(start_index, end_index)
    print("Parsing objects for current range: ", curr_range)

    object_lidar_data, output_dir = parse_objects_from_nuscenes(points_threshold, object_name, object_tag, curr_range)
