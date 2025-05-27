import os
import json
import glob
import random
from collections import defaultdict

def combine_and_split_metadata(metadata_root, output_root, seed=42, train_ratio=0.8):
    os.makedirs(output_root, exist_ok=True)
    by_class_dir = os.path.join(output_root, "by_class")
    os.makedirs(by_class_dir, exist_ok=True)

    # Step 1: gather all metadata
    metadata_files = glob.glob(os.path.join(metadata_root, "**/metadata.json"), recursive=True)
    combined = []
    by_class = defaultdict(list)

    for path in metadata_files:
        with open(path, "r") as f:
            data = json.load(f)
            for obj in data:
                combined.append(obj)
                class_name = obj.get("class_name", "unknown")
                by_class[class_name].append(obj)

    # Step 2: save combined
    with open(os.path.join(output_root, "combined_metadata_corrected_split.json"), "w") as f:
        json.dump(combined, f, indent=2)

    # Step 3: per-class stratified split and save
    for class_name, objects in by_class.items():
        counts = defaultdict(int)
        for item in objects:
            counts[item["sequence"]] += 1

        def split_dict_by_value_weight(d, ratio=0.8):
            total = sum(d.values())
            target = total * ratio

            # Sort keys by descending value for greedy fill
            sorted_items = sorted(d.items(), key=lambda x: x[1], reverse=True)

            train_keys = []
            val_keys = []
            running_sum = 0

            for k, v in sorted_items:
                if running_sum + v <= target:
                    train_keys.append(k)
                    running_sum += v
                else:
                    val_keys.append(k)

            return train_keys, val_keys

        train_keys, val_keys = split_dict_by_value_weight(counts)
        split_data = {"train": [], "val": []}
        for item in objects:
            if item["sequence"] in train_keys:
                split_data["train"].append(item)
            elif item["sequence"] in val_keys:
                split_data["val"].append(item)

        class_dir = os.path.join(by_class_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        with open(os.path.join(class_dir, "instances.json"), "w") as f:
            json.dump(split_data, f, indent=2)

    print(f"✔ Combined {len(combined)} objects from {len(metadata_files)} files")
    print(f"✔ Wrote per-class splits into: {by_class_dir}/<class_name>/train_val_split_by_sequence.json")

if __name__ == "__main__":
    metadata_root = "/home/ekirby/scania/ekirby/datasets/logen_datasets/KITTI-360/processed/objects"
    output_root = "/home/ekirby/scania/ekirby/datasets/logen_datasets/KITTI-360/processed/combined"
    combine_and_split_metadata(metadata_root, output_root)
