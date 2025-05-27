import os
import glob
import json
import click
import yaml
import torch
import numpy as np
import torch.distributed as dist
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.multiprocessing import spawn
from diffusers import DPMSolverMultistepScheduler
from nuscenes.utils.data_classes import Quaternion

from logen.models.diffuser import Diffuser
from logen.datasets.dataset_mapper import dataloaders
from logen.modules.three_d_helpers import cylindrical_to_cartesian, angle_add, estimate_lidar_points_batched

def inverse_scale_intensity(scaled_data):
    max_intensity = np.log1p(255.0)
    data_log_transformed = scaled_data * max_intensity
    return np.round(np.clip(np.expm1(data_log_transformed), a_min=0.0, a_max=255.0))

def realign_pointclouds_to_scan(points, orientation, center, aligned_angle):
    cos_yaw, sin_yaw = np.cos(orientation), np.sin(orientation)
    rotation_matrix = np.array([
        [cos_yaw, -sin_yaw, 0],
        [sin_yaw, cos_yaw, 0],
        [0, 0, 1]
    ])
    points[:, :3] = np.dot(points[:, :3], rotation_matrix.T)
    new_center = center.copy()
    new_center[0] = angle_add(aligned_angle, orientation)
    new_center = cylindrical_to_cartesian(new_center[None, :]).squeeze(0)
    points[:, :3] += new_center
    if points.shape[1] > 3:
        points[:, 3] = inverse_scale_intensity(points[:, 3])
    return points

def generate_new_distances(D_0, num_instances, min_factor=0.5, max_factor=2.0, strategy="linear_uniform"):
    if strategy == "linear_uniform":
        distances = np.linspace(D_0 * min_factor, D_0 * max_factor, num=num_instances)
        return distances + np.random.uniform(-0.1, 0.1, size=distances.shape)
    elif strategy == "log_uniform":
        D_0 = D_0.repeat(num_instances)
        return np.exp(np.random.uniform(np.log(D_0 * min_factor), np.log(D_0 * max_factor), size=(len(D_0) * num_instances)))
    else:
        raise ValueError("Invalid distance generation strategy")

@click.command()
@click.option('--config', '-c', type=str, required=True)
@click.option('--weights', '-w', type=str, required=True)
@click.option('--num_instances', '-n', type=int, default=4)
@click.option('--split', '-s', type=str, default='train')
@click.option('--rootdir', '-r', type=str, required=True)
@click.option('--token_to_data', type=str, required=True)
@click.option('--consistent_seed', type=bool, default=True)
@click.option('--permutation_file', type=str, default=None)
@click.option('--limit_samples_count', type=int, default=-1)
@click.option('--condition', type=str, default='cylinder_angle')
def main(config, weights, num_instances, split, rootdir, token_to_data, consistent_seed, permutation_file, limit_samples_count, condition):
    cfg = yaml.safe_load(open(config))
    world_size = cfg['train']['n_gpus']
    experiment_dir = cfg['experiment']['id']
    existing_paths = glob.glob(f'{rootdir}/{experiment_dir}/*/{cfg["data"]["gen_class_name"]}/**/original_0.txt')
    existing_tokens = set(path.split('/')[-2] for path in existing_paths)

    if world_size > 1:
        spawn(gen, args=(world_size, cfg, weights, num_instances, split, rootdir, cfg['train']['batch_size'], token_to_data, consistent_seed, existing_tokens, permutation_file, limit_samples_count, condition), nprocs=world_size, join=True)
    else:
        gen(0, world_size, cfg, weights, num_instances, split, rootdir, cfg['train']['batch_size'], token_to_data, consistent_seed, existing_tokens, permutation_file, limit_samples_count, condition)

def gen(rank, world_size, cfg, weights, num_instances, split, rootdir, batch_size, token_to_data, consistent_seed, existing_tokens, permutation_file, limit_samples_count, condition):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)

    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    with open(token_to_data, 'r') as f:
        token_to_data_map = json.load(f)[split]

    model = Diffuser.load_from_checkpoint(weights, hparams=cfg, strict=False).to(device)
    model = DDP(model, device_ids=[rank])

    existing_tokens = set()
    permutation = []

    dataset_builder = dataloaders[cfg['data']['dataloader']](cfg)
    dataset, collate = dataset_builder.build_dataset(split, existing_tokens, permutation)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=cfg['train']['num_workers'], collate_fn=collate)

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            model.module.dpm_scheduler = DPMSolverMultistepScheduler(
                num_train_timesteps=model.module.t_steps,
                beta_start=model.module.hparams['diff']['beta_start'],
                beta_end=model.module.hparams['diff']['beta_end'],
                beta_schedule='linear',
                algorithm_type='sde-dpmsolver++',
                solver_order=2
            )
            model.module.dpm_scheduler.set_timesteps(model.module.s_steps)
            model.module.scheduler_to_cuda()
            x_object = batch['pcd_object'].to(device)
            padding_mask = batch['padding_mask'].to(device)
            annotation_tokens = batch['tokens']
            x_class = batch['class'].to(device)
            
            x_center = batch['center']
            x_size = batch['size']
            x_orientation = batch['orientation']
            x_cond = torch.cat((x_center, x_size), dim=-1)

            if condition == "recreation":
                x_t = torch.randn(x_object.shape, device=model.device)
            elif condition == "cylinder_angle": # Interpolate the angle condition around a ring
                starting_angles = x_center[:, 0]
                linspace_ring = np.linspace(start=starting_angles, stop=starting_angles + 2 * np.pi, num=num_instances+1, endpoint=False)
                linspace_ring = (linspace_ring + np.pi) % (2 * np.pi) - np.pi
                linspace_ring = torch.from_numpy(linspace_ring).flatten()
                x_object = x_object.repeat((num_instances+1, 1, 1))
                x_t = torch.randn(x_object.shape, device=model.device)
                x_cond = x_cond.repeat((num_instances+1, 1))
                padding_mask = padding_mask.repeat((num_instances+1, 1))
                x_cond[:, 0] = linspace_ring
            elif condition == "cylinder_distance": # Interpolatethe distance condition according to a power law
                starting_num_points = batch['num_points']
                starting_distances = x_center[:, 1]
                new_distances = torch.from_numpy(generate_new_distances(starting_distances, num_instances, min_factor=0.5, max_factor=2.0, strategy="linear_uniform")).flatten()
                starting_num_points = starting_num_points.repeat((num_instances))
                starting_distances = starting_distances.repeat((num_instances))
                new_num_points = estimate_lidar_points_batched(starting_num_points, starting_distances, new_distances)
                new_max_num_points = new_num_points.max()
                new_objects_tensor = torch.randn((batch_size * num_instances, 4, new_max_num_points))
                old_padding_mask = padding_mask.repeat((num_instances, 1))
                padding_mask = torch.zeros((batch_size * num_instances, new_max_num_points), device=model.device)
                for i in range(batch_size * num_instances):
                    padding_mask[i, :new_num_points[i]] = 1
                x_object = x_object.repeat((num_instances, 1, 1))
                x_t = torch.randn(new_objects_tensor.shape, device=model.device)
                x_cond = x_cond.repeat((num_instances, 1))
                x_cond[:, 1] = new_distances
            else:
                raise ValueError(f"Unknown interpolation condition: {condition}")

            x_cond = x_cond.to(device)
            x_gen = model.module.p_sample_loop(x_t, x_class, x_cond, padding_mask[:, None, :]).permute(0, 2, 1).cpu().numpy()
            x_org = x_object.permute(0, 2, 1).cpu().numpy()

            for i in range(x_gen.shape[0]):
                example_index = i % batch_size
                token = annotation_tokens[example_index]
                sample_data = token_to_data_map[token]

                center = x_center[example_index].cpu().numpy()
                orientation = x_orientation[example_index].cpu().numpy().item()
                gen_mask = padding_mask[i].cpu().bool().numpy()

                if condition == "recreation":
                    x_gen_realigned = realign_pointclouds_to_scan(x_gen[i][gen_mask], orientation, center, center[0])
                    x_org_realigned = realign_pointclouds_to_scan(x_org[i][gen_mask], orientation, center, center[0])
                elif condition == "cylinder_angle":
                    center[0] = linspace_ring[i]
                    x_gen_realigned = realign_pointclouds_to_scan(x_gen[i][gen_mask], orientation, center, linspace_ring[i])
                    x_org_realigned = realign_pointclouds_to_scan(x_org[i][gen_mask], orientation, center, center[0])
                elif condition == "cylinder_distance":
                    old_mask = old_padding_mask[i].cpu().bool().numpy() # Needed because number of points has changed
                    x_org_realigned = realign_pointclouds_to_scan(x_org[i][old_mask], orientation, center, center[0])
                    center_copy = center.copy()
                    center_copy[1] = x_cond[i][1]
                    x_gen_realigned = realign_pointclouds_to_scan(x_gen[i][gen_mask], orientation, center_copy, center[0])

                if 'rotation_real' in sample_data:
                    rotation_real = np.array([sample_data['rotation_real']])
                    rotation_imag = np.array(sample_data['rotation_imaginary'])
                    yaw = Quaternion(real=rotation_real, imaginary=rotation_imag).yaw_pitch_roll[0]
                    condition_data = np.concatenate((center, sample_data['size'], [yaw], rotation_real, rotation_imag))
                    sample_token = sample_data['sample_token']
                else:
                    yaw = sample_data['rotation_yaw']
                    condition_data = np.concatenate((center, sample_data['size'], [yaw]))
                    sample_token = os.path.basename(sample_data['pointcloud_path'])[:-4]

                class_name = cfg['data']['gen_class_name']
                sample_dir = os.path.join(rootdir, cfg['experiment']['id'], sample_token, class_name, token)
                os.makedirs(sample_dir, exist_ok=True)
                idx = i // batch_size
                np.savetxt(os.path.join(sample_dir, f'generated_{idx}.txt'), x_gen_realigned)
                np.savetxt(os.path.join(sample_dir, f'original_{idx}.txt'), x_org_realigned)
                np.savetxt(os.path.join(sample_dir, f'conditions_{idx}.txt'), condition_data)

    dist.destroy_process_group()

if __name__ == '__main__':
    main()
