import os
import numpy as np
import open3d as o3d
import json
from tqdm import tqdm
from collections import defaultdict
import glob
import sklearn
import multiprocessing as mp
import sys

IGNORE_LABEL = 255

id_to_name = {
    24: 'person',
    26: 'car',
    27: 'truck',
    28: 'bus',
    30: 'trailer',
    32: 'motorcycle',
    33: 'bicycle',
}

def extract_objects_from_ply(ply_path, output_dir, sequence_name, min_points=5, is_dynamic=True):
    os.makedirs(output_dir, exist_ok=True)
    pcd = o3d.t.io.read_point_cloud(ply_path)
    pts = pcd.point['positions'].numpy()
    semantic = (pcd.point['semantic']).numpy().astype(np.uint8).squeeze()
    instance = (pcd.point['instance']).numpy().astype(np.uint32).squeeze()
    if is_dynamic:
        timestamps = pcd.point['timestamp'].numpy().astype(np.uint32).squeeze()
    else:
        timestamps = np.ones_like(semantic) * -1

    object_dicts = []
    already_seen = set()
    accepted_classes = id_to_name

    for point_index in range(len(pts)):
        current_set = (semantic[point_index], instance[point_index], timestamps[point_index])
        if current_set in already_seen:
            continue
        already_seen.add(current_set)
        (sem_id, inst_id, ts) = current_set

        if sem_id == IGNORE_LABEL or inst_id == 0 or sem_id not in accepted_classes:
            continue

        mask = (semantic == sem_id) & (instance == inst_id) & (timestamps == ts)
        obj_points = pts[mask]

        if len(obj_points) < min_points:
            continue

        object_id = f"{sequence_name}_{ts:010d}_{inst_id:04d}"
        out_path = os.path.join(output_dir, object_id + ".txt")
        np.savetxt(out_path, obj_points)

        center = np.mean(obj_points, axis=0)
        size = obj_points.max(axis=0) - obj_points.min(axis=0)

        try:
            xy = obj_points[:, :2]
            if len(xy) >= 3 and np.linalg.matrix_rank(xy - xy.mean(0)) >= 2:
                from sklearn.decomposition import PCA
                import warnings
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                pca = PCA(n_components=2).fit(xy)
                direction = pca.components_[0]
                yaw = float(np.arctan2(direction[1], direction[0]))
            else:
                yaw = 0.0
        except RuntimeWarning:
            continue

        object_dicts.append({
            "sequence": sequence_name,
            "frame": int(ts),
            "instance_id": int(inst_id),
            "semantic_class": int(sem_id),
            "class_name": accepted_classes[sem_id],
            "num_points": int(len(obj_points)),
            "pointcloud_path": out_path,
            "center": center.tolist(),
            "size": size.tolist(),
            "rotation_yaw": yaw
        })

    return object_dicts


def process_ply_file(args):
    ply_path, is_dynamic, base_output_dir = args
    sequence_name = os.path.basename(ply_path)[:-4]
    output_dir = os.path.join(base_output_dir, sequence_name)
    print("Parsing", sequence_name)
    obj_data = extract_objects_from_ply(ply_path, output_dir, sequence_name, is_dynamic=is_dynamic)
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(obj_data, f, indent=2)
    return len(obj_data)


if __name__ == '__main__':
    static_ply_paths = glob.glob(sys.argv[1]) # Something like "KITTI-360/data_3d_semantics/train/*/static/*.ply"
    dynamic_ply_paths = glob.glob(sys.argv[2]) # Something like "KITTI-360/data_3d_semantics/train/*/dynamic/*.ply"
    output_dir = sys.argv[3] # Something like KITTI-360/processed/train/objects/
    all_tasks = [(p, False, output_dir) for p in static_ply_paths] + [(p, True, output_dir) for p in dynamic_ply_paths]

    with mp.Pool(processes=os.cpu_count()) as pool:
        obj_counts = list(tqdm(pool.imap(process_ply_file, all_tasks), total=len(all_tasks)))

    print("Parsed total objects:", sum(obj_counts))
