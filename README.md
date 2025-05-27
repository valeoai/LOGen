# Official Repository of "LOGen: Towards LiDAR Object Generation by Point Diffusion"
[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/abs/2412.07385)


## Installation

Requirements:
```
pip install torch
pip install timm
pip install pytorch-lightning
pip install diffusers
pip install nuscenes-devkit
pip install einops
pip install ninja
pip install scipy
pip install open3d
pip install matplotlib
pip install trimesh
pip install Ninja
pip install modules/PyTorchEMD
pip install -e .
```

To install `pytorch3d` (used in evaluation code) follow instructions from https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md

## Datasets
All data processing scripts are located in `data_preprocess`
### Nuscenes
The Nuscenes object dataset consists of a set of json files, one per object class. The json files record the IDs, conditions, and required paths to retreive an object from the Nuscenes dataset. These objects were parsed using a series of helper scripts. 
- `parse_objects_from_nuscenes.py` is designed to take a range of objects and extract the necessary information to use them in the dataloaders. *Important:* the current set-up of this script extracts only objects with at least 20 points. These objects are cleaned, meaning that they consist of only points with the canonical class label (i.e. no points labeled ground or anything else). This can be configured in the script, but the current datasets were created with this constraint.
- `parse_instance_masks.py` is a second script which iterates through all extracted objects and creates a boolean mask within the overall LiDAR scene. This allows us to quickly find the object in the scene.
- Finally a simple python script was used to convert the json lists into maps, where the key is the sample annotation token. This allows us to easily filter objects by their split when generating.
### KITTI-360
The KITTI-360 data is similarly structured to the extracted nuscenes objects, the but the extraction process is simpler. 
- `parse_kitti.py` reads static and dynamic `.ply` files from the KITTI-360 dataset and uses instance masks to extract objects by class per scene.
- `merge_and_split_kitti.py` merges objects by class and then conducts the train/val split on a per-scene level. We do not have semantic labels for the KITTI-360 validation set so we use this script to build our own.

## Training a model
All models use Pytorch-lightning to run training and inference. In order to conduct the numerous ablation studies, configuration files are used to control which specific model will be used.

`train.py` contains the code to launch the Pytorch-lightning modules for both training and inference. This script uses tensorboard logging to record training, validation, and testing events, and logs all outputs to the `experiments/experiment_id` directory. Checkpoints are saved to `checkpoints/experiment_id/last.ckpt`. 

Some notes about the default training:
- Multi GPU is handled via the config file where `['train']['n_gpus']` is set. 
- By default the training script first runs two validation iterations to perform a sanity check.
- When validating, only two batches are used by default, because validation requires a full diffusion process. 
- For the same reason, in all training configs the `s_steps` parameter is set to 50, to allow faster evaluation.
- Running the script with the `--test` flag will evaluate the FPD and CD across the entire validation set, using 50 diffusion timesteps.
## Generating Data

We use seperate config files for generating data because of variances in the batch size and the number of GPUs. 

We have a script for multi-gpu training, `gen_instance_pool_multi_gpu.py`, which will check for existing files and generate according to the passed config. Alternatively a permutation file can be specified that and a specific number of instances can be generated.

## Frechet Pointnet Distance

We have trained custom vesions of the Pointnet classifier using the script at `scripts/train_pnet.sh`. Here are some details on the organization of this part of the project.

`evaluation/fpd` is the folder that contains the dataloaders and training loops to train the Pointnet model used for the FPD evaluation. `pretrained` was used for the evaluation that uses the Modelnet trained Pointnet. `from_scratch` contains the code for the Pointnet model trained on Nuscenes objects.

There are different checkpoints to use depending on which type of data being evaluated, as the dataset the Pointnet was trained on has a large impact on the final FPD. Here are the important checkpoints, stored in `evaluation/fpd/from_scratch/checkpoints`:
- `evaluation/fpd/from_scratch/checkpoints/cleaned_nuscenes_objects`: This is a Pointnet classifier trained on the 10 classes of the cleaned Nuscenes objects dataset, with 4 channels as input.
- `evaluation/fpd/from_scratch/checkpoints/cleaned_nuscenes_objects_3ch`: This is a Pointnet classifier trained on the 10 classes of the cleaned Nuscenes objects dataset, but with 3 channels as input.

Both of these models were trained for 50 epochs.