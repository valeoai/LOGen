import click
import torch
from evaluation.CD_EMD.paired_dataloader import NuscenesPairedObjectsDataLoader
# from modules.metrics import EMD
from logen.modules.PyTorchEMD.emd import earth_mover_distance as EMD
# from modules.metrics import ChamferDistance
import os
import numpy as np
import open3d as o3d
from logen.modules.three_d_helpers import build_two_point_clouds
# from ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import chamfer_3DDist
# from ChamferDistancePytorch.fscore import fscore
from pytorch3d.loss import chamfer_distance
try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

def get_data(dataloader):
    all_generated_points = []
    all_real_points = []
    for i, batch in tqdm(enumerate(dataloader)):

        generated_points = batch[0]
        real_points = batch[1]

        all_generated_points.append(generated_points)
        all_real_points.append(real_points)


    return all_generated_points, all_real_points

# cham3D = chamfer_3DDist()


def cham3D(gens, reals):

    generated_points = gens.squeeze(0)[:, :3]
    real_points = reals.squeeze(0)[:, :3]

    pcd_pred, pcd_gt = build_two_point_clouds(genrtd_pcd=generated_points, object_pcd=real_points)
    pt_pcd = pcd_pred
    gt_pcd = pcd_gt

    dist_pt_2_gt = np.asarray(pt_pcd.compute_point_cloud_distance(gt_pcd))
    dist_gt_2_pt = np.asarray(gt_pcd.compute_point_cloud_distance(pt_pcd))

    final_dist = (np.mean(dist_gt_2_pt) + np.mean(dist_pt_2_gt)) / 2
    return dist_pt_2_gt, dist_gt_2_pt, final_dist


def _pairwise_CD_(sample_pcs, ref_pcs):
    N_sample = len(sample_pcs)
    N_ref = len(ref_pcs)
    all_cd = []

    iterator = range(N_sample)
    batch_size = N_ref
    for sample_b_start in tqdm(iterator):

        sample_batch = sample_pcs[sample_b_start]

        cd_lst = []

        for ref_b_start in range(0, N_ref, batch_size):
            ref_b_end = min(N_ref, ref_b_start + batch_size)
            ref_batch = ref_pcs[ref_b_start:ref_b_end]
            batch_size_ref = len(ref_batch)
            main_point_nums = [sample_batch.shape[1]] * batch_size_ref
            x_lengths = torch.tensor(main_point_nums, dtype=torch.long).cuda()

            point_nums = [tensor.size(1) for tensor in ref_batch]
            y_lengths = torch.tensor(point_nums, dtype=torch.long).cuda()
            max_rows = max(point_nums)
            padded_data = [torch.nn.functional.pad(tensor, (0, 0, 0, max_rows - tensor.size(1)), "constant", 100) for tensor in ref_batch]
            padded_data = torch.stack(padded_data).squeeze()

            sample_batch_exp = sample_batch.view(1, -1, 3).expand(batch_size_ref, -1, -1)
            sample_batch_exp = sample_batch_exp.contiguous()

            cd_dists, _ = chamfer_distance(sample_batch_exp.cuda(), padded_data.cuda(), x_lengths=x_lengths, y_lengths=y_lengths, batch_reduction=None)
            cd_lst.append(cd_dists.view(1, -1).detach().cpu())

        cd_lst = torch.cat(cd_lst, dim=1)
        all_cd.append(cd_lst)

    all_cd = torch.cat(all_cd, dim=0)  # N_sample, N_ref

    return all_cd


# def _pairwise_EMD_CD_(sample_pcs, ref_pcs, batch_size, accelerated_cd=True):
#     N_sample = len(sample_pcs) #sample_pcs.shape[0]
#     N_ref = len(ref_pcs) #ref_pcs.shape[0]
#     all_cd = []
#     all_emd = []
#     iterator = range(N_sample)
#
#     batch_size = 1
#     for sample_b_start in tqdm(iterator):
#
#         sample_batch = sample_pcs[sample_b_start]
#
#         cd_lst = []
#         emd_lst = []
#         for ref_b_start in tqdm(range(0, N_ref, batch_size)):
#             ref_b_end = min(N_ref, ref_b_start + batch_size)
#             ref_batch = ref_pcs[ref_b_start:ref_b_end]
#             batch_size_ref = len(ref_batch)
#             main_point_nums = [sample_batch.shape[1]] * batch_size_ref
#             x_lengths = torch.tensor(main_point_nums, dtype=torch.long).cuda()
#             # assert len(ref_batch) > 1
#             # ref_batch = ref_batch[0]
#             # ref_batch2 = ref_pcs[0:3]
#             point_nums = [tensor.size(1) for tensor in ref_batch]
#             y_lengths = torch.tensor(point_nums, dtype=torch.long).cuda()
#             max_rows = max(point_nums)
#             padded_data = [torch.nn.functional.pad(tensor, (0, 0, 0, max_rows - tensor.size(1)), "constant", 100) for tensor in ref_batch]
#             padded_data = torch.stack(padded_data).squeeze()
#
#             sample_batch_exp = sample_batch.view(1, -1, 3).expand(batch_size_ref, -1, -1)
#             sample_batch_exp = sample_batch_exp.contiguous()
#
#             cd_dists = EMD(sample_batch_exp.cuda(), padded_data.cuda(), transpose=False)
#             cd_lst.append(cd_dists.view(1, -1).detach().cpu())
#
#             # dl, dr, _, _ = cham3D(sample_batch_exp.cuda(), ref_batch.cuda())
#             # cd_dist = (dl.mean(dim=1) + dr.mean(dim=1)).view(1, -1).detach().cpu()
#             # cd_lst.append(cd_dist)
#
#             # cd_dists, _ = chamfer_distance(sample_batch_exp.cuda(), padded_data.cuda(), x_lengths=x_lengths, y_lengths=y_lengths, batch_reduction=None)
#             # cd_lst.append(cd_dists.view(1, -1).detach().cpu())
#
#             # ddist = cham3D(sample_batch_exp.cuda(), ref_batch.cuda())
#             # ddist = torch.zeros_like(emd_batch) + ddist
#             # ddist = ddist
#             # cd_lst.append(ddist.view(1, -1).detach().cpu())
#             # dl, dr, _, _ = cham3D(sample_batch_exp.cuda(), ref_batch.cuda())
#             # cd_lst.append((dl.mean(dim=1) + dr.mean(dim=1)).view(1, -1).detach().cpu())
#
#         cd_lst = torch.cat(cd_lst, dim=1)
#         # emd_lst = torch.cat(emd_lst, dim=1)
#         all_cd.append(cd_lst)
#         # all_emd.append(emd_lst)
#
#     all_cd = torch.cat(all_cd, dim=0)  # N_sample, N_ref
#     # all_emd = torch.cat(all_emd, dim=0)  # N_sample, N_ref
#
#     return all_cd, all_cd #all_cd


def _pairwise_EMD(sample_pcs, ref_pcs, batch_size=1):

    N_sample = len(sample_pcs)
    N_ref = len(ref_pcs)
    all_emd = []
    iterator = range(N_sample)
    for sample_b_start in tqdm(iterator):
        sample_batch = sample_pcs[sample_b_start]
        emd_lst = []
        for ref_b_start in range(0, N_ref, batch_size):
            ref_b_end = min(N_ref, ref_b_start + batch_size)
            ref_batch = ref_pcs[ref_b_start:ref_b_end]
            ref_batch = ref_batch[0]

            batch_size_ref = ref_batch.size(0)
            sample_batch_exp = sample_batch.view(1, -1, 3).expand(batch_size_ref, -1, -1)
            sample_batch_exp = sample_batch_exp.contiguous()

            emd_batch = EMD(sample_batch_exp.cuda(), ref_batch.cuda(), transpose=False)
            emd_lst.append(emd_batch.view(1, -1).detach().cpu())

        emd_lst = torch.cat(emd_lst, dim=1)
        all_emd.append(emd_lst)

    all_emd = torch.cat(all_emd, dim=0)  # N_sample, N_ref

    return all_emd

def lgan_mmd_cov(all_dist):
    N_sample, N_ref = all_dist.size(0), all_dist.size(1)
    min_val_fromsmp, min_idx = torch.min(all_dist, dim=1)
    min_val, _ = torch.min(all_dist, dim=0)
    mmd = min_val.mean()
    mmd_smp = min_val_fromsmp.mean()
    cov = float(min_idx.unique().view(-1).size(0)) / float(N_ref)
    cov = torch.tensor(cov).to(all_dist)
    return {
        'lgan_mmd': mmd,
        'lgan_cov': cov,
        'lgan_mmd_smp': mmd_smp,
    }

def knn(Mxx, Mxy, Myy, k, sqrt=False):
    n0 = Mxx.size(0)
    n1 = Myy.size(0)
    label = torch.cat((torch.ones(n0), torch.zeros(n1))).to(Mxx)
    M = torch.cat((torch.cat((Mxx, Mxy), 1), torch.cat((Mxy.transpose(0, 1), Myy), 1)), 0)
    if sqrt:
        M = M.abs().sqrt()
    INFINITY = float('inf')
    val, idx = (M + torch.diag(INFINITY * torch.ones(n0 + n1).to(Mxx))).topk(k, 0, False)

    count = torch.zeros(n0 + n1).to(Mxx)
    for i in range(0, k):
        count = count + label.index_select(0, idx[i])
    pred = torch.ge(count, (float(k) / 2) * torch.ones(n0 + n1).to(Mxx)).float()

    s = {
        'tp': (pred * label).sum(),
        'fp': (pred * (1 - label)).sum(),
        'fn': ((1 - pred) * label).sum(),
        'tn': ((1 - pred) * (1 - label)).sum(),
    }

    s.update({
        'precision': s['tp'] / (s['tp'] + s['fp'] + 1e-10),
        'recall': s['tp'] / (s['tp'] + s['fn'] + 1e-10),
        'acc_t': s['tp'] / (s['tp'] + s['fn'] + 1e-10),
        'acc_f': s['tn'] / (s['tn'] + s['fp'] + 1e-10),
        'acc': torch.eq(label, pred).float().mean(),
    })
    return s

def get_1nn_cov(model_name, root, split, object_class, input_channels, distance_method):

    print(f'Evaluating: {model_name} {split} on distance method {distance_method}')
    rootdir = root + '/' + model_name
    dl = NuscenesPairedObjectsDataLoader(root=rootdir, split=split, input_channels=input_channels, object_class=object_class)
    dl = torch.utils.data.DataLoader(dl, batch_size=1, shuffle=False, pin_memory=True, num_workers=0)

    # device = torch.device('cuda:0')
    # sample_pcs = torch.rand((2, 5, 3))
    # ref_pcs = torch.rand((2, 5, 3))
    all_generated_points, all_real_points = get_data(dl)
    sample_pcs = all_generated_points
    ref_pcs = all_real_points

    results = {}

    if distance_method=="CD":
        M_rs_cd = _pairwise_CD_(ref_pcs, sample_pcs)
        res_cd = lgan_mmd_cov(M_rs_cd.t())
        results.update({
            "%s-CD" % k: v for k, v in res_cd.items()
        })
        M_rr_cd = _pairwise_CD_(ref_pcs, ref_pcs)
        M_ss_cd = _pairwise_CD_(sample_pcs, sample_pcs)
        one_nn_cd_res = knn(M_rr_cd, M_rs_cd, M_ss_cd, 1, sqrt=False)
        results.update({
            "1-NN-CD-%s" % k: v for k, v in one_nn_cd_res.items() if 'acc' in k
        })
    elif distance_method=="EMD":
        M_rs_emd = _pairwise_EMD(ref_pcs, sample_pcs, batch_size=1)
        res_emd = lgan_mmd_cov(M_rs_emd.t())
        results.update({
            "%s-EMD" % k: v for k, v in res_emd.items()
        })
        M_rr_emd = _pairwise_EMD(ref_pcs, ref_pcs, batch_size=1)
        M_ss_emd = _pairwise_EMD(sample_pcs, sample_pcs, batch_size=1)
        one_nn_emd_res = knn(M_rr_emd, M_rs_emd, M_ss_emd, 1, sqrt=False)
        results.update({
            "1-NN-EMD-%s" % k: v for k, v in one_nn_emd_res.items() if 'acc' in k
        })

    results = {k: (v.cpu().detach().item()
                   if not isinstance(v, float) else v) for k, v in results.items()}

    os.makedirs(f'./evaluation/1NN_COV/experiments/{model_name}', exist_ok=True)
    with open(f'./evaluation/1NN_COV/experiments/{model_name}/{distance_method}_{split}_{object_class}.txt', 'w') as f:
        for kk in results:
            print(f'{kk}: {results[kk]} ', file=f)

    print(f'lgan_cov-CD: {results["lgan_cov-CD"]} ')
    print(f'1-NN-CD-acc: {results["1-NN-CD-acc"]} ')

@click.command()
@click.option('--model_name', '-m', type=str, default='xs_4_1a_cross_pointnet_impcgf_4chfix_reordered_gen_2')
@click.option('--root', '-r', type=str)
@click.option('--split', '-s', type=str, default='train')
@click.option('--object_class', '-cls', type=str)
@click.option('--input_channels', '-i', type=int, default=4)
@click.option('--distance_method', '-d', type=str, default="CD")
def main(model_name, root, split, object_class, input_channels, distance_method):
    get_1nn_cov(model_name, root, split, object_class, input_channels, distance_method)

if __name__ == "__main__":
    main()


