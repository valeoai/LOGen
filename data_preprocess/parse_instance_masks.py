import numpy as np
from nuscenes import NuScenes
from nuscenes.utils.data_classes import Quaternion, Box
from nuscenes.utils.geometry_utils import points_in_box
import json
from tqdm import tqdm
import click
import os

def extract_semantic_instance_ids(nusc, samples_file, split, class_name):
    with open(samples_file, 'r') as r:
        samples = json.load(r)[split]
    
    new_samples = []
    for sample in tqdm(samples):
        if class_name in sample['class']:
            lidarseg_label_filename = os.path.join(nusc.dataroot, nusc.get('lidarseg', sample['sample_data_lidar_token'])['filename'])
            labels = np.fromfile(lidarseg_label_filename, dtype=np.uint8).reshape((-1, 1))
            points = np.fromfile(sample['lidar_data_filepath'],  dtype=np.float32).reshape((-1, 5))
            center = np.array(sample['center'])
            size = np.array(sample['size'])
            rotation_real = np.array(sample['rotation_real'])
            rotation_imaginary = np.array(sample['rotation_imaginary'])

            orientation = Quaternion(real=rotation_real, imaginary=rotation_imaginary)
            box = Box(center=center, size=size, orientation=orientation)

            points_from_object = points_in_box(box, points=points[:,:3].T)
            object_isolated_label_mask = (points_from_object & (labels == nusc.lidarseg_name2idx_mapping[sample['class']]).flatten())
            filename = f'{sample["instance_token"]}.npy'
            np.save(filename, object_isolated_label_mask)
            sample['object_sample_index'] = filename
            new_samples.append(sample)
    
    return new_samples


@click.command
@click.option('--instances_file', '-f', type=str) # Input objects
@click.option('--save_file', '-s', type=str)
@click.option('-class_name', '-cls', type=str)
def main(instances_file, save_file, class_name):
    dataroot = '/datasets_local/nuscenes'

    # Initialize the nuScenes class
    nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True)
    samples_with_indices_train = extract_semantic_instance_ids(nusc, instances_file, 'train', class_name)
    samples_with_indices_val   = extract_semantic_instance_ids(nusc, instances_file, 'val', class_name)
    fixed_samples = {'train':samples_with_indices_train, 'val':samples_with_indices_val}
    tokens_to_data_remapping = {'train':{}, 'val':{}}
    for split in fixed_samples.keys():
        for sample in fixed_samples[split]:
            tokens_to_data_remapping[split][sample['instance_token']] = sample
            
    with open(save_file, 'w') as w:
        json.dump(tokens_to_data_remapping, w)

if __name__=='__main__':
    main()
