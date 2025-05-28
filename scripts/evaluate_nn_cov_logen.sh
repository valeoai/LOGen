# !/bin/bash
eval_model=nuscenes
split=val
root_dir=augmented_nuscenes_datasets
channels=3
distance_method=EMD

# Compute BIKES
model=xs_1a_logen_bikes_cleaned_gen_999
python compute_1nn_cov.py -m $model -s $split -i $channels -r $root_dir  -cls bike -d $distance_method

# Compute BUS
model=xs_1a_logen_bus_gen_999
python compute_1nn_cov.py -m $model -s $split -i $channels -r $root_dir -cls bus -d $distance_method

# Compute MOTORCYCLE
model=xs_1a_logen_motorcycles_gen_999
python compute_1nn_cov.py -m $model -s $split -i $channels -r $root_dir -cls motorcycle -d $distance_method

# Compute PEDESTRIAN
model=xs_1a_logen_pedestrian_gen_999
python compute_1nn_cov.py -m $model -s $split -i $channels -r $root_dir -cls pedestrian -d $distance_method

# Compute TRAFFIC CONE
model=xs_1a_logen_traffic_cones_gen_999
python compute_1nn_cov.py -m $model -s $split -i $channels -r $root_dir -cls traffic_cone -d $distance_method

# Compute TRUCKS
model=xs_1a_logen_trucks_gen_999
python compute_1nn_cov.py -m $model -s $split -i $channels -r $root_dir -cls truck -d $distance_method

# Compute CONSTRUCTION VEHICLE
model=xs_1a_logen_construction_vehicles_gen_999
python compute_1nn_cov.py -m $model -s $split -i $channels -r $root_dir -cls construction_vehicle -d $distance_method

# Compute TRAILER
model=xs_1a_logen_trailers_gen_999
python compute_1nn_cov.py -m $model -s $split -i $channels -r $root_dir -cls trailer -d $distance_method

# Compute BARRIERS
model=xs_1a_logen_barriers_gen_999
python compute_1nn_cov.py -m $model -s $split -i $channels -r $root_dir -cls barrier -d $distance_method

# Compute CAR
model=xs_1a_logen_cars_gen_999
python compute_1nn_cov.py -m $model -s $split -i $channels -r $root_dir -cls car -d $distance_method


