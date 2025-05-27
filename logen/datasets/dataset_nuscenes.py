import torch
import torch.nn.functional
import torch.utils
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
import warnings
import numpy as np
from logen.datasets.dataloader_nuscenes import NuscenesObjectsSet

warnings.filterwarnings('ignore')

__all__ = ['NuscenesObjectsDataModule']

class NuscenesObjectsDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def prepare_data(self):
        # Augmentations
        pass

    def setup(self, stage=None):
        # Create datasets
        pass
    
    def build_dataset(self, split, excluded_tokens=None, permutation=None):
        collate = NuscenesObjectCollator()

        data_set = NuscenesObjectsSet(
                data_dir=self.cfg['data']['data_dir'], 
                split=split, 
                input_channels=self.cfg['model']['in_channels'],
                excluded_tokens=excluded_tokens,
                permutation=permutation
            )
        
        return data_set, collate

    def train_dataloader(self, shuffle=True):
        collate = NuscenesObjectCollator()

        data_set = NuscenesObjectsSet(
                data_dir=self.cfg['data']['data_dir'], 
                split='train', 
                input_channels=self.cfg['model']['in_channels']
            )
        loader = DataLoader(data_set, batch_size=self.cfg['train']['batch_size'], shuffle=shuffle,
                            num_workers=self.cfg['train']['num_workers'], collate_fn=collate)
        return loader

    def val_dataloader(self, pre_training=True):
        collate = NuscenesObjectCollator()

        data_set = NuscenesObjectsSet(
                data_dir=self.cfg['data']['data_dir'], 
                split='val',
                input_channels=self.cfg['model']['in_channels']
            )
        loader = DataLoader(data_set, batch_size=self.cfg['train']['batch_size'], shuffle=True,
                            num_workers=self.cfg['train']['num_workers'], collate_fn=collate)
        return loader

    def test_dataloader(self):
        collate = NuscenesObjectCollator()

        data_set = NuscenesObjectsSet(
                data_dir=self.cfg['data']['data_dir'], 
                split='val', 
                input_channels=self.cfg['model']['in_channels']
            )
        loader = DataLoader(data_set, batch_size=self.cfg['train']['batch_size'],
                             num_workers=self.cfg['train']['num_workers'], collate_fn=collate)
        return loader

class NuscenesObjectCollator:
    def __call__(self, data):
        batch = list(zip(*data))

        num_points_tensor = torch.Tensor(batch[4])

        pcd_object = batch[0]
        input_channels = pcd_object[0].shape[1]
        max_points = num_points_tensor.max().int().item()
        padded_point_clouds = np.zeros((len(pcd_object), max_points, input_channels))
        padding_mask = np.zeros((len(pcd_object), max_points))
        for i, pc in enumerate(pcd_object):
            padded_point_clouds[i, :pc.shape[0], :] = pc
            padding_mask [i, :pc.shape[0]] = 1

        pcd_object = torch.from_numpy(padded_point_clouds).float()
        padding_mask = torch.from_numpy(padding_mask).float()
        
        pcd_object = pcd_object.permute(0,2,1)
        batch_indices = torch.zeros(len(pcd_object))

        return {'pcd_object': pcd_object, 
            'center':  torch.from_numpy(np.vstack(batch[1])).float(),
            'size':  torch.from_numpy(np.vstack(batch[2])).float(),
            'orientation': torch.from_numpy(np.vstack(batch[3])).float(),
            'batch_indices': batch_indices,
            'num_points': num_points_tensor,
            'class': torch.vstack(batch[5]),
            'padding_mask': padding_mask,
            'tokens': batch[7]
        }

dataloaders = {
    'nuscenes': NuscenesObjectsDataModule,
}

