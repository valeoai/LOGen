from torch.utils.data import Dataset
import numpy as np
import glob
import json

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

class NuscenesPairedObjectsDataLoader(Dataset):
    def __init__(self, root, split, input_channels, object_class):
        super().__init__()
        paths = glob.glob(f'{root}/*/{object_class}/**')
        self.input_channels = input_channels
        # with open('/home/nsamet/scania/ekirby/datasets/augmented_nuscenes_datasets/nuscenes_scene_token_to_split.json', 'r') as f:
        #     self.split_mapping = json.load(f)[split]
        # self.dirs = [path for path in paths if path.split('/')[-3] in self.split_mapping]
        self.dirs = paths

    def __len__(self):
        return len(self.dirs)
    
    def __getitem__(self, index):
        curr_dir = self.dirs[index]
        real_object_data = np.loadtxt(f'{curr_dir}/original_0.txt', dtype=np.float32)
        generated_object_data = np.loadtxt(f'{curr_dir}/generated_0.txt', dtype=np.float32)
        label = 1
        real_points = real_object_data[:, :self.input_channels]
        real_points = pc_normalize_axiswise(real_points)
        generated_points = generated_object_data[:, :self.input_channels]
        generated_points = pc_normalize_axiswise(generated_points)
        return generated_points, real_points, label

