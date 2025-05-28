import click
import numpy as np
import torch
from evaluation.fpd.pretrained.pointnet import PointNetCls
from evaluation.fpd.from_scratch.pointnet_cls import Pointnet as NSPointnet
from evaluation.fpd.from_scratch.nuscenes_fpd_dataloader import num_classes
from evaluation.fpd.pretrained.dataloader import NuscenesGeneratedObjectsDataLoader
# from evaluation.fpd.FPD import calculate_kid
import os
try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x): return x

from torch import randint
from torchmetrics.image.kid import KernelInceptionDistance

def get_kid(model_name, root, split, object_class, evaluation_model, input_channels, pointnet_path, class_label, real):
    print(f'Evaluating: {model_name} {split}')
    if real:
        print(f'Evaluating GTs.')
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
        model = NSPointnet(k=num_classes, input_channels=input_channels, return_act=False)
        model.load_state_dict(new_state_dict)
    
    if device is not None:
        model.to(device)

    model.eval()

    if real:
        curr_dataloader=dl_real
    else:
        curr_dataloader=dl_generated

    correct = 0
    total = 0
    all_pred = []
    for i, batch in tqdm(enumerate(curr_dataloader)):
        pointcloud_batch = batch[0]
        if device is not None:
            pointcloud_batch = pointcloud_batch.to(device)
        pointcloud_batch = pointcloud_batch.transpose(1, 2)
        output, _, _ = model(pointcloud_batch)
        _, pred = output.data.cpu().topk(1, dim=1)
        all_pred.append(pred[0][0].data.cpu().numpy() )
        if pred[0][0].data.cpu().numpy() == class_label:
            correct+=1
        total += 1

    a, b = np.unique(np.asarray(all_pred), return_counts=True)
    print(a)
    print(b)
    print(f"Correct instance number: {correct}")
    print(f"Total instance number: {total}")
    print(f"Accuracy is  {correct/total}")

@click.command()
@click.option('--model_name', '-m', type=str, default='xs_4_1a_cross_pointnet_impcgf_4chfix_reordered_gen_2')
@click.option('--root', '-r', type=str)
@click.option('--split', '-s', type=str, default='train')
@click.option('--object_class', '-cls', type=str)
@click.option('--evaluation_model', '-e', type=str,)
@click.option('--input_channels', '-i', type=int, default=4)
@click.option('--pointnet_checkpoint_path', '-pckpt', type=str)
@click.option('--class_label',   type=int, default=0)
@click.option('--real', type=bool, default=False) # default False
def main(model_name, root, split, object_class, evaluation_model, input_channels, pointnet_checkpoint_path, class_label, real):
    get_kid(model_name, root, split, object_class, evaluation_model, input_channels, pointnet_checkpoint_path, class_label, real)

if __name__ == "__main__":
    main()
