# !/bin/bash

eval_model=nuscenes
split=val
root_dir=replaced_nuscenes_datasets

model_type=$1 # xs_1a_dit3d_pe_4ch
channels=$2

if [[ "$channels" -eq 4 ]]; then
pnet_ckpt=evaluation/fpd/from_scratch/checkpoints/cleaned_nuscenes_objects/last.ckpt
fi

if [[ "$channels" -eq 3 ]]; then
pnet_ckpt=evaluation/fpd/from_scratch/checkpoints/cleaned_nuscenes_objects_3ch/last.ckpt
fi

echo $pnet_ckpt

# Compute MOTORCYCLE
model=${model_type}_motorcycles_gen_999
python compute_kid.py -m $model -s $split -e $eval_model -cls motorcycle -i $channels -pckpt $pnet_ckpt

# Compute BARRIERS
model=${model_type}_barriers_gen_999
python compute_kid.py -m $model -s $split -e $eval_model -cls barrier -i $channels -pckpt $pnet_ckpt

# Compute BIKES
model=${model_type}_bikes_cleaned_gen_999
python compute_kid.py -m $model -s $split -e $eval_model -cls bike -i $channels -pckpt $pnet_ckpt

# Compute BUS
model=${model_type}_bus_gen_999
python compute_kid.py -m $model -s $split -e $eval_model -cls bus -i $channels -pckpt $pnet_ckpt

# Compute PEDESTRIAN
model=${model_type}_pedestrian_gen_999
python compute_kid.py -m $model -s $split -e $eval_model -cls pedestrian -i $channels -pckpt $pnet_ckpt

# Compute TRAFFIC CONE
model=${model_type}_traffic_cones_gen_999
python compute_kid.py -m $model -s $split -e $eval_model -cls traffic_cone -i $channels -pckpt $pnet_ckpt

# Compute TRUCKS
model=${model_type}_trucks_gen_999
python compute_kid.py -m $model -s $split -e $eval_model -cls truck -i $channels -pckpt $pnet_ckpt

# Compute CAR
model=${model_type}_cars_gen_999
python compute_kid.py -m $model -s $split -e $eval_model -cls car -i $channels -pckpt $pnet_ckpt

# Compute CONSTRUCTION VEHICLE
model=${model_type}_construction_vehicles_gen_999
python compute_kid.py -m $model -s $split -e $eval_model -cls construction_vehicle -i $channels -pckpt $pnet_ckpt
#
# Compute TRAILER
model=${model_type}_trailers_gen_999
python compute_kid.py -m $model -s $split -e $eval_model -cls trailer -i $channels -pckpt $pnet_ckpt
