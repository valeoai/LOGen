import click
import torch
from evaluation.fpd.pretrained.pointnet import PointNetCls
from evaluation.fpd.from_scratch.pointnet_cls import Pointnet as NSPointnet
from evaluation.fpd.from_scratch.nuscenes_fpd_dataloader import num_classes
from evaluation.fpd.pretrained.dataloader import NuscenesGeneratedObjectsDataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
import os
try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x): return x


def get_fid(model_name, root, split, object_class, evaluation_model, input_channels, pointnet_path):
    print(f'Evaluating: {model_name} {split}')
    rootdir = root + '/' + model_name

    dl_generated = NuscenesGeneratedObjectsDataLoader(root=rootdir, real_or_generated='generated', split=split, num_points=1024, input_channels=input_channels, object_class=object_class)
    dl_generated = torch.utils.data.DataLoader(dl_generated, batch_size=1, shuffle=False, pin_memory=True, num_workers=4)
    dl_real =      NuscenesGeneratedObjectsDataLoader(root=rootdir, real_or_generated='real', split=split, num_points=1024, input_channels=input_channels, object_class=object_class)
    dl_real =      torch.utils.data.DataLoader(dl_real, batch_size=1, shuffle=False, pin_memory=True, num_workers=4)

    device = torch.device('cuda:0')

    if evaluation_model == 'modelnet':
        PointNet_path = 'evaluation/fpd/pretrained/cls_model_39.pth'
        model = PointNetCls(k=16)
        model.load_state_dict(torch.load(PointNet_path))
    elif evaluation_model == 'nuscenes':
        PointNet_path = pointnet_path
        checkpoint = torch.load(PointNet_path)
        # Adjust the keys in the state dictionary
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for key, value in state_dict.items():
            # Remove the 'model.' prefix
            new_key = key.replace('model.', '', 1)
            new_state_dict[new_key] = value
        model_ns = NSPointnet(k=num_classes, input_channels=input_channels, return_act=True)
        model_ns.load_state_dict(new_state_dict)
    
    if device is not None:
        model_ns.to(device)

    model_ns.eval()

    fid = FrechetInceptionDistance(feature=model_ns).to(device)

    for i, batch in tqdm(enumerate(dl_generated)):
        pointcloud_batch = batch[0]
        if device is not None:
            pointcloud_batch = pointcloud_batch.to(device)
        pointcloud_batch = pointcloud_batch.transpose(1, 2)
        fid.update(pointcloud_batch, real=False)

    for i, batch in tqdm(enumerate(dl_real)):
        pointcloud_batch = batch[0]
        if device is not None:
            pointcloud_batch = pointcloud_batch.to(device)
        pointcloud_batch = pointcloud_batch.transpose(1, 2)
        fid.update(pointcloud_batch, real=True)

    fid_score = fid.compute()

    print('Frechet Inception Distance <<< {:.10f} >>>'.format(fid_score))
    os.makedirs(f'./evaluation/fid/experiments_distance_gens_05x_{input_channels}ch/{model_name}', exist_ok=True)
    with open(f'./evaluation/fid/experiments_distance_gens_05x_{input_channels}ch/{model_name}/fid_{evaluation_model}_{split}_{object_class}.txt', 'w') as f:
        print('Frechet Pointcloud Distance <<< {:.10f} >>>'.format(fid_score), file=f)


@click.command()
@click.option('--model_name', '-m', type=str, default='xs_logen_kitti360_car_gen_split_by_sequence')
@click.option('--root', '-r', type=str)
@click.option('--split', '-s', type=str, default='val')
@click.option('--object_class', '-cls', type=str, default='car')
@click.option('--evaluation_model', '-e', type=str, default='nuscenes')
@click.option('--input_channels', '-i', type=int, default=3)
@click.option('--pointnet_checkpoint_path', '-pckpt', type=str)
def main(model_name, root, split, object_class, evaluation_model, input_channels, pointnet_checkpoint_path):
    get_fid(model_name, root, split, object_class, evaluation_model, input_channels, pointnet_checkpoint_path)

if __name__ == "__main__":
    main()

