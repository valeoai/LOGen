#!/bin/bash

# # Gen Instances
config=config/logen_config.yaml
weights=checkpoints/logen_bikes_last.ckpt
tokens_to_data=/path/to/tokens_to_data_mapping.json
python generation/gen_instance_pool.py -c $config -w $weights -n 3 -s val -r /tmp/ --token_to_data $tokens_to_data --condition cylinder_angle
