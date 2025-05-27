import os
import yaml
import click
import torch
import numpy as np
import open3d as o3d
from tqdm import tqdm
from diffusers import DPMSolverMultistepScheduler

from logen.models.diffuser import Diffuser
from logen.modules.three_d_helpers import cylindrical_to_cartesian, angle_add, estimate_lidar_points_batched
from logen.datasets import dataset_mapper
from logen.modules.class_mapping import class_mapping

def inverse_scale_intensity(scaled_data):
    max_intensity = np.log1p(255.0)
    data_log_transformed = scaled_data * max_intensity
    return np.round(np.clip(np.expm1(data_log_transformed), a_min=0.0, a_max=255.0))

def realign_pointclouds_to_scan(x_gen, orientation, center, aligned_angle):
    cos_yaw, sin_yaw = np.cos(orientation), np.sin(orientation)
    rotation_matrix = np.array([
        [cos_yaw, -sin_yaw, 0],
        [sin_yaw, cos_yaw, 0],
        [0, 0, 1]
    ])
    x_gen[:, :3] = np.dot(x_gen[:, :3], rotation_matrix.T)
    new_center = center.copy()
    new_center[0] = angle_add(aligned_angle, orientation)
    new_center = cylindrical_to_cartesian(new_center[None, :]).squeeze(0)
    x_gen[:, :3] += new_center
    x_gen[:, 3] = inverse_scale_intensity(x_gen[:, 3])
    return x_gen

def generate_new_distances(D_0, num_instances, min_factor=0.5, max_factor=2.0, strategy="linear_uniform"):
    if strategy == "linear_uniform":
        D_x = np.linspace(D_0 * min_factor, D_0 * max_factor, num=num_instances)
        D_x += np.random.uniform(-0.1, 0.1, size=D_x.shape)
    elif strategy == "log_uniform":
        D_0 = D_0.repeat(num_instances)
        log_min, log_max = np.log(D_0 * min_factor), np.log(D_0 * max_factor)
        D_x = np.exp(np.random.uniform(log_min, log_max, size=(len(D_0) * num_instances)))
    else:
        raise ValueError("Invalid strategy. Choose 'linear_uniform' or 'log_uniform'.")
    return D_x

def find_eligible_objects(dataloader, num_to_find=1, object_class='vehicle.car', min_points=None, target_id=None):
    targets = []
    for index, item in enumerate(dataloader):
        item['index'] = index
        if target_id != None and item['tokens'][0] == target_id:
            targets.append(item)
            print("Found target")
            break
        if object_class != 'None' and class_mapping[object_class] != item['class'].item():
            continue
        if min_points and item['num_points'][0] > min_points:
            targets.append(item)
        if len(targets) >= num_to_find:
            break
    return targets

def find_specific_objects(index, cfg):
    module = dataset_mapper.dataloaders[cfg['data']['dataloader']](cfg)
    dataloader = module.train_dataloader(shuffle=False)
    for i, item in enumerate(dataloader):
        if i == index:
            item['index'] = index
            return [item]
    return []

def visualize_step_t(x_t, pcd):
    points = x_t.detach().cpu().numpy()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def p_sample_loop(model, x_t, x_cond, x_class, viz_path=None):
    scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=model.t_steps,
        beta_start=model.hparams['diff']['beta_start'],
        beta_end=model.hparams['diff']['beta_end'],
        beta_schedule='linear',
        algorithm_type='sde-dpmsolver++',
        solver_order=2
    )
    scheduler.set_timesteps(model.s_steps)
    model.dpm_scheduler = scheduler
    model.scheduler_to_cuda()
    viz_pcd = o3d.geometry.PointCloud() if viz_path else None

    for step, t in enumerate(tqdm(scheduler.timesteps)):
        t = t.cuda()[None]
        with torch.no_grad():
            noise_t = model.classfree_forward(x_t, t, x_class, x_cond)
        x_t = scheduler.step(noise_t, t, x_t)['prev_sample']

        if viz_path:
            viz = visualize_step_t(x_t.clone(), viz_pcd)
            o3d.io.write_point_cloud(f'{viz_path}/step_visualizations/step_{step:03d}.ply', viz)

    return x_t

def denoise_object(model, x_object, x_center, x_size, x_orientation, x_class, num_samples, viz_path=None):
    x_cond = torch.cat((x_center, x_size, x_orientation), dim=-1).cuda()
    x_class = x_class.cuda()
    x_object = x_object.cuda()
    results = []

    for _ in range(num_samples):
        x_feats = torch.randn_like(x_object)
        x_gen = p_sample_loop(model, x_feats, x_cond, x_class, viz_path=viz_path).squeeze(0).permute(1, 0)
        results.append(x_gen[:, :4].cpu().numpy())

    return results

@click.command()
@click.option('--config', '-c', type=str, required=True, help='Path to the config file (.yaml)')
@click.option('--weights', '-w', type=str, required=True, help='Path to pretrained weights (.ckpt)')
@click.option('--output_path', '-o', type=str, default='generated_objects', help='Output directory')
@click.option('--name', '-n', type=str, default='experiment', help='Subfolder name for outputs')
@click.option('--task', '-t', type=str, default='recreate', help='Task to run: recreate or interpolate')
@click.option('--class_name', '-cls', type=str, default='vehicle.car', help='Class label to generate')
@click.option('--split', '-s', type=str, default='train', help='Dataset split to use')
@click.option('--min_points', '-m', type=int, default=100, help='Minimum points required')
@click.option('--do_viz', '-v', is_flag=True, help='Enable step visualizations')
@click.option('--examples_to_generate', '-e', type=int, default=1, help='Number of examples to generate')
@click.option('--num_samples', '-ds', type=int, default=10, help='Number of diffusion samples per object')
@click.option('--specific_obj_index', '-ind', type=int, default=None, help='Specific object index to use')
def main(config, weights, output_path, name, task, class_name, split, min_points, do_viz, examples_to_generate, num_samples, specific_obj_index):
    cfg = yaml.safe_load(open(config))
    cfg['diff']['s_steps'] = 999
    cfg['train']['batch_size'] = 1
    model = Diffuser.load_from_checkpoint(weights, hparams=cfg, strict=False).cuda().eval()
    dir_path = os.path.join(output_path, name)
    os.makedirs(dir_path, exist_ok=True)

    if specific_obj_index is not None:
        objects = find_specific_objects(specific_obj_index, cfg)
    else:
        module = dataset_mapper.dataloaders[cfg['data']['dataloader']](cfg)
        dataloader = module.train_dataloader() if split == 'train' else module.val_dataloader()
        objects = find_eligible_objects(dataloader, num_to_find=examples_to_generate, object_class=class_name, min_points=min_points)

    for obj in objects:
        x_object = obj['pcd_object']
        x_center = obj['center']
        x_size = obj['size']
        x_orientation = obj['orientation']
        x_class = obj['class']

        gen_results = denoise_object(model, x_object, x_center, x_size, x_orientation, x_class, num_samples, viz_path=dir_path if do_viz else None)
        for i, gen in enumerate(gen_results):
            np.savetxt(os.path.join(dir_path, f"{obj['index']}_sample_{i}.txt"), gen)

if __name__ == '__main__':
    main()
