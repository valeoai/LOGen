import torch
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
import warnings
import numpy as np
from logen.datasets.dataloader_kitti360 import Kitti360ObjectsSet

warnings.filterwarnings('ignore')

__all__ = ['Kitti360ObjectsDataModule']

class Kitti360ObjectsDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass

    def build_dataset(self, split, excluded_ids=None, permutation=None):
        collate = Kitti360ObjectCollator(self.cfg['data']['stacking_type'])
        dataset = Kitti360ObjectsSet(
            data_dir=self.cfg['data']['data_dir'],
            split=split,
            points_per_object=self.cfg['data']['points_per_object'],
            align_objects=self.cfg['data']['align_objects'],
            relative_angles=self.cfg['model']['relative_angles'],
            stacking_type=self.cfg['data']['stacking_type'],
            class_conditional=self.cfg['train']['class_conditional'],
            normalize_points=self.cfg['data']['normalize_points'],
            input_channels=self.cfg['model']['in_channels'],
            excluded_ids=excluded_ids,
        )
        return dataset, collate

    def train_dataloader(self, shuffle=True):
        dataset, collate = self.build_dataset('train')
        return DataLoader(dataset, batch_size=self.cfg['train']['batch_size'], shuffle=shuffle,
                          num_workers=self.cfg['train']['num_workers'], collate_fn=collate)

    def val_dataloader(self):
        dataset, collate = self.build_dataset('val')
        return DataLoader(dataset, batch_size=self.cfg['train']['batch_size'], shuffle=False,
                          num_workers=self.cfg['train']['num_workers'], collate_fn=collate)

    def test_dataloader(self):
        dataset, collate = self.build_dataset('val')
        return DataLoader(dataset, batch_size=self.cfg['train']['batch_size'], shuffle=False,
                          num_workers=self.cfg['train']['num_workers'], collate_fn=collate)

class Kitti360ObjectCollator:
    def __init__(self, stacking_type):
        self.max_stack = stacking_type == 'max'

    def __call__(self, data):
        batch = list(zip(*data))
        num_points_tensor = torch.Tensor(batch[4])

        if self.max_stack:
            pcd_object = batch[0]
            input_channels = pcd_object[0].shape[1]
            max_points = num_points_tensor.max().int().item()
            padded_point_clouds = np.zeros((len(pcd_object), max_points, input_channels))
            padding_mask = np.zeros((len(pcd_object), max_points))
            for i, pc in enumerate(pcd_object):
                padded_point_clouds[i, :pc.shape[0], :] = pc
                padding_mask[i, :pc.shape[0]] = 1
            pcd_object = torch.from_numpy(padded_point_clouds).float()
            padding_mask = torch.from_numpy(padding_mask).float()
        else:
            pcd_object = torch.from_numpy(np.stack(batch[0]))
            padding_mask = torch.from_numpy(np.stack(batch[6]))

        pcd_object = pcd_object.permute(0, 2, 1)
        batch_indices = torch.zeros(len(pcd_object))

        return {
            'pcd_object': pcd_object,
            'center': torch.from_numpy(np.vstack(batch[1])).float(),
            'size': torch.from_numpy(np.vstack(batch[2])).float(),
            'orientation': torch.from_numpy(np.vstack(batch[3])).float(),
            'batch_indices': batch_indices,
            'num_points': num_points_tensor,
            'class': torch.vstack(batch[5]),
            'padding_mask': padding_mask,
            'tokens': batch[7]
        }

dataloaders = {
    'kitti360': Kitti360ObjectsDataModule,
}
