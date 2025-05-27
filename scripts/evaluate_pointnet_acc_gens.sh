# !/bin/bash

eval_model=nuscenes
split=val
root_dir=/home/nsamet/scania/nsamet/replaced_nuscenes_datasets
model_type=$1 #

# channels=4
# pnet_ckpt=/home/nsamet/no_backup/repos/LiDAR-Object-Generation/evaluation/fpd/from_scratch/checkpoints/cleaned_nuscenes_objects/last.ckpt

channels=3
pnet_ckpt=/home/nsamet/no_backup/repos/LiDAR-Object-Generation/evaluation/fpd/from_scratch/checkpoints/cleaned_nuscenes_objects_3ch/last.ckpt

# Compute BARRIERS
model=${model_type}_barriers_gen_999
python compute_classification_acc.py -m $model -s $split -e $eval_model -cls barrier -i $channels -pckpt $pnet_ckpt --class_label 1

# Compute BIKES
model=${model_type}_bikes_cleaned_gen_999
python compute_classification_acc.py -m $model -s $split -e $eval_model -cls bike -i $channels -pckpt $pnet_ckpt --class_label 4

# Compute BUS
model=${model_type}_bus_gen_999
python compute_classification_acc.py -m $model -s $split -e $eval_model -cls bus -i $channels -pckpt $pnet_ckpt --class_label 5

# Compute MOTORCYCLE
model=${model_type}_motorcycles_gen_999
python compute_classification_acc.py -m $model -s $split -e $eval_model -cls motorcycle -i $channels -pckpt $pnet_ckpt --class_label 8

# Compute PEDESTRIAN
model=${model_type}_pedestrian_gen_999
python compute_classification_acc.py -m $model -s $split -e $eval_model -cls pedestrian -i $channels -pckpt $pnet_ckpt --class_label 0

# Compute TRAFFIC CONE
model=${model_type}_traffic_cones_gen_999
python compute_classification_acc.py -m $model -s $split -e $eval_model -cls traffic_cone -i $channels -pckpt $pnet_ckpt --class_label 3

# Compute TRUCKS
model=${model_type}_trucks_gen_999
python compute_classification_acc.py -m $model -s $split -e $eval_model -cls truck -i $channels -pckpt $pnet_ckpt --class_label 10

# Compute CONSTRUCTION VEHICLE
model=${model_type}_construction_vehicles_gen_999
python compute_classification_acc.py -m $model -s $split -e $eval_model -cls construction_vehicle -i $channels -pckpt $pnet_ckpt --class_label 7

# Compute TRAILER
model=${model_type}_trailers_gen_999
python compute_classification_acc.py -m $model -s $split -e $eval_model -cls trailer -i $channels -pckpt $pnet_ckpt --class_label 9

# Compute CAR
model=${model_type}_cars_gen_999
python compute_classification_acc.py -m $model -s $split -e $eval_model -cls car -i $channels -pckpt $pnet_ckpt --class_label 6