import click
import torch
from evaluation.CD_EMD.paired_dataloader import NuscenesPairedObjectsDataLoader
import os
import scipy
import scipy.spatial
import numpy as np
from evaluation.CD_EMD.CD_EMD import calculate_cd_emd

class JSD():
    def __init__(self):
        self.dists = []

        return

    def update(self, gt_np, pt_np):
        self.dists.append(scipy.spatial.distance.jensenshannon(gt_np.cpu().detach().numpy(), pt_np.cpu().detach().numpy()).mean())

    def reset(self):
        self.dists = []

    def compute(self):
        cdist = np.array(self.dists)
        # return cdist.mean(), cdist.std()
        return cdist[np.isfinite(cdist)].mean(),  cdist[np.isfinite(cdist)].std() 
    
    def last_cd(self):
        return self.dists[-1]
    
    def best_index(self):
        return np.array(self.dists).argmin()  
    



def get_jsd(model_name, root, split, object_class, input_channels):
    print(f'Evaluating: {model_name} {split}', flush=True)
    rootdir = root + '/' + model_name
    dl = NuscenesPairedObjectsDataLoader(root=rootdir, split=split, input_channels=input_channels, object_class=object_class)
    dl = torch.utils.data.DataLoader(dl, batch_size=1, shuffle=False, pin_memory=True, num_workers=0)

    device = torch.device('cuda:0')

    # cd = ChamferDistance()

    jsd = JSD()
    jsd_score = calculate_cd_emd(dl, jsd, device=device, needs_pointclouds=False)
    # jsd.update(p, n)
    # jsd_score = jsd.compute()

    print('JSD mean <<< {:.10f} >>> and std <<< {:.10f} >>> '.format(jsd_score[0], jsd_score[1]))

 
    # os.makedirs(f'./evaluation/JSD/experiments/{model_name}', exist_ok=True)
    # with open(f'./evaluation/JSD/experiments/{model_name}/cd_emd_{split}_{object_class}.txt', 'w') as f:
    #     print('JSD mean <<< {:.10f} >>> and std <<< {:.10f} >>> '.format(cd_score[0], cd_score[1]), file=f) 


@click.command()
@click.option('--model_name', '-m', type=str, default='xs_logen_kitti360_bicycle_gen_split_by_sequence')
@click.option('--root', '-r', type=str, default='/home/nsamet/scania/ekirby/logen-experiments/KITTI-360/bikes_gen_split_by_sequence')
@click.option('--split', '-s', type=str, default='val')
@click.option('--object_class', '-cls', type=str, default='bicycle')
@click.option('--input_channels', '-i', type=int, default=3)
def main(model_name, root, split, object_class, input_channels):
    get_jsd(model_name, root, split, object_class, input_channels)

if __name__ == "__main__":
    main()


# @click.command()
# @click.option('--model_name', '-m', type=str, default='xs_4_1a_cross_pointnet_impcgf_4chfix_reordered_gen_2')
# @click.option('--root', '-r', type=str, default='/home/nsamet/scania/ekirby/datasets/augmented_nuscenes_datasets')
# @click.option('--split', '-s', type=str, default='train')
# @click.option('--object_class', '-cls', type=str)
# @click.option('--input_channels', '-i', type=int, default=4)