import os
import pathlib
import numpy as np
import torch
from torch import Tensor
from modules.three_d_helpers import build_two_point_clouds

try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x): return x

def get_scores(dataloader, metric, needs_pointclouds, device=None, verbose=False):
    for i, batch in tqdm(enumerate(dataloader)):
        if verbose:
            print('\rPropagating batch %d/%d' % (i + 1, len(dataloader)),
                  end='', flush=True)

        generated_points = batch[0]
        real_points = batch[1]
        
        if device is not None:
            generated_points = generated_points.to(device).squeeze(0)[:,:3]
            real_points = real_points.to(device).squeeze(0)[:,:3]
        
        if needs_pointclouds:
            pcd_pred, pcd_gt = build_two_point_clouds(genrtd_pcd=generated_points, object_pcd=real_points)
            metric.update(pcd_gt, pcd_pred)
        else:
            metric.update(real_points, generated_points)
        
    if verbose:
        print(' done')

    return metric.compute()


def calculate_cd_emd(dl, metric, device=None, needs_pointclouds=True):
    """Calculates the FPD of two pointclouds"""
    score  = get_scores(dl, metric, needs_pointclouds, device)
    return score
