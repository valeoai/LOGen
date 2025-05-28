import click
import torch
from evaluation.CD_EMD.paired_dataloader import NuscenesPairedObjectsDataLoader
from evaluation.CD_EMD.CD_EMD import calculate_cd_emd
from logen.modules.metrics import ChamferDistance, EMD
import os

def get_cd_emd(model_name, root, split, object_class, input_channels):
    print(f'Evaluating: {model_name} {split}', flush=True)
    rootdir = root + '/' + model_name
    dl = NuscenesPairedObjectsDataLoader(root=rootdir, split=split, input_channels=input_channels, object_class=object_class)
    dl = torch.utils.data.DataLoader(dl, batch_size=1, shuffle=False, pin_memory=True, num_workers=0)

    device = torch.device('cuda:0')

    cd = ChamferDistance()
    cd_score = calculate_cd_emd(dl, cd, device=device, needs_pointclouds=True)

    print('CD mean <<< {:.10f} >>> and std <<< {:.10f} >>> '.format(cd_score[0], cd_score[1]))

    emd = EMD()
    emd_score = calculate_cd_emd(dl, emd, device=device, needs_pointclouds=False)
    print('EMD mean <<< {:.10f} >>> and std <<< {:.10f} >>>'.format(emd_score[0], emd_score[1]))

    os.makedirs(f'./evaluation/CD_EMD/experiments/{model_name}', exist_ok=True)
    with open(f'./evaluation/CD_EMD/experiments/{model_name}/cd_emd_{split}_{object_class}.txt', 'w') as f:
        print('CD mean <<< {:.10f} >>> and std <<< {:.10f} >>> '.format(cd_score[0], cd_score[1]), file=f)
        print('EMD mean <<< {:.10f} >>> and std <<< {:.10f} >>>'.format(emd_score[0], emd_score[1]), file=f)


@click.command()
@click.option('--model_name', '-m', type=str, default='xs_logen_kitti360_bicycle_gen_split_by_sequence')
@click.option('--root', '-r', type=str)
@click.option('--split', '-s', type=str, default='val')
@click.option('--object_class', '-cls', type=str, default='bicycle')
@click.option('--input_channels', '-i', type=int, default=3)
def main(model_name, root, split, object_class, input_channels):
    get_cd_emd(model_name, root, split, object_class, input_channels)

if __name__ == "__main__":
    main()
