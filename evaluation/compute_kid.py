import click
import torch
from evaluation.fpd.pretrained.pointnet import PointNetCls
from evaluation.fpd.from_scratch.pointnet_cls import Pointnet as NSPointnet
from evaluation.fpd.from_scratch.pointnet_cls import SPVCNN as NSspvcnn
from evaluation.fpd.from_scratch.nuscenes_fpd_dataloader import num_classes
from evaluation.fpd.pretrained.dataloader import NuscenesGeneratedObjectsDataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
import os
try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x): return x

from torch import randint
from torchmetrics.image.kid import KernelInceptionDistance

def get_kid(model_name, root, split, object_class, evaluation_model, input_channels, pointnet_path):
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
        model = NSPointnet(k=num_classes, input_channels=input_channels, return_act=True)
        model.load_state_dict(new_state_dict)
    elif evaluation_model == 'spvcnn':
        print("Evaluation on SPVCNN features")
        splitted = model_name.split("_")
        if (object_class == "bike") or (object_class == "traffic_cone") or (object_class == "construction_vehicle"):
            gen_model = '_'.join(splitted[:-4])
        else:
            gen_model = '_'.join(splitted[:-3])
        model = NSspvcnn(gen_model,input_channels=input_channels)

    if device is not None:
        model.to(device)

    model.eval()

    kid = KernelInceptionDistance(feature=model, subset_size=50)

    for i, batch in tqdm(enumerate(dl_generated)):
        if evaluation_model == 'nuscenes':
            pointcloud_batch = batch[0]
            if device is not None:
                pointcloud_batch = pointcloud_batch.to(device)
            pointcloud_batch = pointcloud_batch.transpose(1, 2)
            kid.update(pointcloud_batch, real=False)
        elif evaluation_model == 'spvcnn':
            inp = (batch[2], "gen")
            kid.update(inp, real=False)

    for i, batch in tqdm(enumerate(dl_real)):

        if evaluation_model == 'nuscenes':
            pointcloud_batch = batch[0]
            if device is not None:
                pointcloud_batch = pointcloud_batch.to(device)
            pointcloud_batch = pointcloud_batch.transpose(1, 2)
            kid.update(pointcloud_batch, real=True)
        elif evaluation_model == 'spvcnn':
            inp = (batch[2], "real")
            kid.update(inp, real=True)

    kid_mean, kid_std  = kid.compute()

    print('Kernel Pointcloud Distance <<< {:.10f} {:.10f} >>>'.format(kid_mean,kid_std))
    os.makedirs(f'./evaluation/kid/experiments_distance_gens_05x_{input_channels}ch/{model_name}', exist_ok=True)
    with open(f'./evaluation/kid/experiments_distance_gens_05x_{input_channels}ch/{model_name}/kid_{evaluation_model}_{split}_{object_class}.txt', 'w') as f:
        print('Kernel Pointcloud Distance <<< {:.10f} >>>'.format(kid_mean), file=f)

@click.command()
@click.option('--model_name', '-m', type=str, default='xs_logen_kitti360_bicycle_gen_split_by_sequence')
@click.option('--root', '-r', type=str)
@click.option('--split', '-s', type=str, default='val')
@click.option('--object_class', '-cls', type=str, default='bicycle')
@click.option('--evaluation_model', '-e', type=str, default='nuscenes')
@click.option('--input_channels', '-i', type=int, default=3)
@click.option('--pointnet_checkpoint_path', '-pckpt', type=str)
def main(model_name, root, split, object_class, evaluation_model, input_channels, pointnet_checkpoint_path):
    get_kid(model_name, root, split, object_class, evaluation_model, input_channels, pointnet_checkpoint_path)

if __name__ == "__main__":
    main()


