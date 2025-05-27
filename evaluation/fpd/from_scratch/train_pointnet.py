import os
import sys
import torch
import argparse
from .model_pointnet import PointnetModule
from .nuscenes_fpd_dataloader import NuscenesObjectsDataModule
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import yaml


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size in training')
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--epoch', default=50, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--weights', default=None)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--ckpt_path', type=str, default=None, help='Checkpoint for resuming training.')
    parser.add_argument('--test_config', default=None)
    parser.add_argument('--num_input_channels', default=None, type=int)
    return parser.parse_args()


def main(args):
    #Load data and model
    if args.weights is None:
        model = PointnetModule(input_channels=args.num_input_channels)
    else:
        model = PointnetModule
        model = model.load_from_checkpoint(args.weights)

    if args.test_config != None:
        cfg = yaml.safe_load(open(args.test_config))
        experiment_id = f'{cfg["experiment_id"]}_{cfg["objects_to_test"]}_{cfg["test_split"]}'
        experiment_root_dir = cfg['experiment_root_dir']
        objects_to_test = cfg['objects_to_test']
        test_split = cfg['test_split']
        filtered_instance_ids = cfg['filtered_instance_ids']
        batch_size = cfg['batch_size']
        class_name = cfg['class_name']
        data = NuscenesObjectsDataModule(
            data_dir=experiment_root_dir,
            num_workers=args.num_workers,
            batch_size=batch_size,
            data_to_use=objects_to_test,
            test_split=test_split,
            filtered_instance_ids=filtered_instance_ids,
            class_name=class_name
        )
    else:
        data = NuscenesObjectsDataModule(
            data_dir='/home/nsamet/scania/ekirby/datasets/all_objects_nuscenes_cleaned/all_objects_nuscenes_cleaned_train_val.json',
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            data_to_use='original_data',
            input_channels=args.num_input_channels,
        )
        experiment_id = 'training'

    #Add callbacks
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_saver = ModelCheckpoint(
                                dirpath='/home/nsamet/no_backup/repos/LiDAR-Object-Generation/evaluation/fpd/from_scratch/checkpoints/cleaned_nuscenes_objects_3ch',
                                filename='{epoch:02d}',
                                save_last=True,
                            )

    tb_logger = pl_loggers.TensorBoardLogger(f'/home/nsamet/no_backup/repos/LiDAR-Object-Generation/evaluation/fpd/from_scratch/experiments/{experiment_id}',
                                             default_hp_metric=False)
    #Setup trainer
    if torch.cuda.device_count() > 1:
        trainer = Trainer(
                        devices=torch.cuda.device_count(),
                        logger=tb_logger,
                        log_every_n_steps=100,
                        max_epochs= args.epoch,
                        callbacks=[lr_monitor, checkpoint_saver],
                        check_val_every_n_epoch=10,
                        num_sanity_val_steps=1,
                        accelerator='gpu',
                        strategy="ddp",
                        )
    else:
        trainer = Trainer(
                        accelerator='gpu',
                        devices=1,
                        logger=tb_logger,
                        log_every_n_steps=100,
                        max_epochs= args.epoch,
                        callbacks=[lr_monitor, checkpoint_saver],
                        check_val_every_n_epoch=10,
                        num_sanity_val_steps=1,
                        )


    # Train!
    if args.test_config != None:
        print('TESTING MODE')
        trainer.test(model, data)
    else:
        print('TRAINING MODE')
        trainer.fit(model, data, ckpt_path=args.ckpt_path)

if __name__ == '__main__':
    args = parse_args()
    main(args)
