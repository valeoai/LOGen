#!/bin/bash

# # Gen Instances
config=config/logen_config.yaml
weights=checkpoints/logen_bikes_last.ckpt
tokens_to_data=/home/ekirby/scania/ekirby/datasets/logen_datasets/all_objects_nuscenes_cleaned/all_objects_cleaned_token_to_data.json
python generation/gen_instance_pool.py -c $config -w $weights -n 3 -s val -r /home/ekirby/tmp/ --token_to_data $tokens_to_data --condition cylinder_angle