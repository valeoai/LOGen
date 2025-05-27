import torch
import torch.nn.functional
import torch.utils
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
import warnings
import numpy as np
from logen.datasets.dataloader_shapenet import ShapeNetObjectsSet

warnings.filterwarnings('ignore')

__all__ = ['ShapeNetObjectsDataModule']

class ShapeNetObjectsDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def prepare_data(self):
        # Augmentations
        pass

    def setup(self, stage=None):
        # Create datasets
        pass

    def train_dataloader(self):
        collate = ShapeNetObjectCollator()

        data_set = ShapeNetObjectsSet(
                data_dir=self.cfg['data']['data_dir'], 
                split='train',
                points_per_object=self.cfg['data']['points_per_object'],
                conditions_data_dir=self.cfg['data']['conditions_dir'],
                relative_angles=self.cfg['model']['relative_angles'],
            )
        loader = DataLoader(data_set, batch_size=self.cfg['train']['batch_size'], shuffle=True,
                            num_workers=self.cfg['train']['num_workers'], collate_fn=collate)
        return loader

    def val_dataloader(self, pre_training=True):
        collate = ShapeNetObjectCollator()

        data_set = ShapeNetObjectsSet(
                data_dir=self.cfg['data']['data_dir'], 
                split='val',
                points_per_object=self.cfg['data']['points_per_object'],
                conditions_data_dir=self.cfg['data']['conditions_dir'],
                relative_angles=self.cfg['model']['relative_angles'],
            )
        loader = DataLoader(data_set, batch_size=self.cfg['train']['batch_size'],
                            num_workers=self.cfg['train']['num_workers'], collate_fn=collate)
        return loader

    def test_dataloader(self):
        collate = ShapeNetObjectCollator()

        data_set = ShapeNetObjectsSet(
                data_dir=self.cfg['data']['data_dir'], 
                split='val',
                points_per_object=self.cfg['data']['points_per_object'],
                conditions_data_dir=self.cfg['data']['conditions_dir'],
                relative_angles=self.cfg['model']['relative_angles'],
            )
        loader = DataLoader(data_set, batch_size=self.cfg['train']['batch_size'],
                             num_workers=self.cfg['train']['num_workers'], collate_fn=collate)
        return loader

class ShapeNetObjectCollator:
    def __init__(self):
        return

    def __call__(self, data):
        # "transpose" the  batch(pt, ptn) to batch(pt), batch(ptn)
        batch = list(zip(*data))
        pcd_object = torch.from_numpy(np.stack(batch[0]))
        pcd_object = pcd_object.permute(0,2,1)

        batch_indices = torch.zeros(pcd_object.shape[0])
        num_points_tensor = torch.Tensor(batch[4])

        return {'pcd_object': pcd_object, 
            'center':  torch.from_numpy(np.vstack(batch[1])).float(),
            'size':  torch.from_numpy(np.vstack(batch[2])).float(),
            'orientation': torch.from_numpy(np.vstack(batch[3])).float(),
            'batch_indices': batch_indices,
            'num_points': num_points_tensor,
            'class': torch.ones((pcd_object.shape[0], 1)).long(),
            'padding_mask': torch.ones((pcd_object.shape[0], pcd_object.shape[2])),
            'tokens': batch[7]
        }

dataloaders = {
    'shapenet': ShapeNetObjectsDataModule,
}

