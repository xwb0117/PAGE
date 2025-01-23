import torch
from utils import get_config, get_log_dir, str2bool
from dataset.dataloader_page import get_loader
from train_PAGE_Estimator import Trainer
import warnings
from tensorboardX import SummaryWriter
import argparse

warnings.filterwarnings('ignore')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--mode',
                        type=str,
                        default='train',
                        choices=['train', 'test'])
    parser.add_argument("--gpu_id", type=int, default=-1)
    parser.add_argument('--dname',
                        type=str,
                        default='Art',
                        choices=['Art', 'ReArt'])
    parser.add_argument("--root_dataset",
                        type=str,
                        default='./mnt/7797b2ec-a944-4795-abb2-f472a7fc833e/Data/ArtImage')
    parser.add_argument("--resume_train", type=str2bool, default=False)
    parser.add_argument("--optim",
                        type=str,
                        default='Adam',
                        choices=['Adam', 'SGD'])
    parser.add_argument("--batch_size",
                        type=str,
                        default='4')
    parser.add_argument("--class_name",
                        type=str,
                        default='laptop')
    parser.add_argument("--max_epoch",
                        type=int,
                        default=300)
    parser.add_argument("--initial_lr",
                        type=float,
                        default=1e-4)
    parser.add_argument('--model_dir',
                        type=str,
                        default='ckpts/')
    parser.add_argument('--demo_mode',
                        type=bool,
                        default=False)
    parser.add_argument('--test_occ',
                        type=bool,
                        default=False)
    parser.add_argument('--using_ckpts',
                        type=bool,
                        default=False)
    parser.add_argument('--part_num',
                        type=int,
                        default=0)
    parser.add_argument('--kpt_num',
                        type=int,
                        default=3)
    parser.add_argument('--kpt_class',
                        type=str,
                        default='KP')
    parser.add_argument('--num_classes',
                        type=int,
                        default=0)
    parser.add_argument('--n_sample_points',
                        type=int,
                        default=2048)
    parser.add_argument('--input_channel',
                        type=int,
                        default=0)
    parser.add_argument('--log_dir',
                        type=str,
                        default='laptop_0')
    parser.add_argument('--kpts_path',
                        type=str,
                        default='laptop_0_mean_points.npy')
    parser.add_argument('--model',
                        type=str,
                        default='KPFusion')
    parser.add_argument('--choose',
                        type=str,
                        default='loss')
    parser.add_argument('--optimize',
                        type=bool,
                        default=False)
    opts = parser.parse_args()

    cfg = get_config()[1]
    opts.cfg = cfg

    if opts.mode in ['train']:
        opts.out = get_log_dir(opts.dname + '/' + opts.log_dir, cfg)
        opts.kpts_path = get_log_dir(f'logs_kp/ArtImage/radii/laptop_{opts.part_num}', opts.kpts_path)
        print('Output logs: ', opts.out)
        vis = SummaryWriter(logdir=opts.out + '/tbLog/')
    else:
        vis = []

    data = get_loader(opts)

    trainer = Trainer(data, opts, vis)
    if opts.mode == 'test':
        trainer.Test()
    else:
        trainer.Train()