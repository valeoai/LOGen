# !/bin/bash

cd evaluation

eval_model=nuscenes
split=val
root_dir=augmented_nuscenes_datasets
channels=3

# Compute BIKES
model=xs_1a_logen_bikes_cleaned_gen_999
python compute_cd_emd.py -m $model -s $split -i $channels -r $root_dir  -cls bike

# Compute BARRIERS
model=xs_1a_logen_barriers_gen_999
python compute_cd_emd.py -m $model -s $split -i $channels -r $root_dir -cls barrier

# Compute BUS
model=xs_1a_logen_bus_gen_999
python compute_cd_emd.py -m $model -s $split -i $channels -r $root_dir -cls bus

# Compute MOTORCYCLE
model=xs_1a_logen_motorcycles_gen_999
python compute_cd_emd.py -m $model -s $split -i $channels -r $root_dir -cls motorcycle

# Compute PEDESTRIAN
model=xs_1a_logen_pedestrian_gen_999
python compute_cd_emd.py -m $model -s $split -i $channels -r $root_dir -cls pedestrian

# Compute TRAFFIC CONE
model=xs_1a_logen_traffic_cones_gen_999
python compute_cd_emd.py -m $model -s $split -i $channels -r $root_dir -cls traffic_cone

# Compute TRUCKS
model=xs_1a_logen_trucks_gen_999
python compute_cd_emd.py -m $model -s $split -i $channels -r $root_dir -cls truck

# Compute CAR
model=xs_1a_logen_cars_gen_999
python compute_cd_emd.py -m $model -s $split -i $channels -r $root_dir -cls car

# Compute CONSTRUCTION VEHICLE
model=xs_1a_logen_construction_vehicles_gen_999
python compute_cd_emd.py -m $model -s $split -i $channels -r $root_dir -cls construction_vehicle

# Compute TRAILER
model=xs_1a_logen_trailers_gen_999
python compute_cd_emd.py -m $model -s $split -i $channels -r $root_dir -cls trailer
