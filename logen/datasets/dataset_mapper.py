from logen.datasets.dataset_nuscenes import NuscenesObjectsDataModule
from logen.datasets.dataset_shapenet import ShapeNetObjectsDataModule
from logen.datasets.dataset_kitti360 import Kitti360ObjectsDataModule

dataloaders = {
    'nuscenes': NuscenesObjectsDataModule,
    'shapenet': ShapeNetObjectsDataModule,
    'kitti360': Kitti360ObjectsDataModule
}
