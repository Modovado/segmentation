from __future__ import annotations
import os
import yaml
os.environ["CUDA_LAUNCH_BLOCKING"] = 'True'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import argparse
import numpy as np
import torch
from torch.utils.data import Subset, DataLoader
from utils.utils import seed_everything, test

from losses.losses import CELoss, DiceLoss
from losses.focal_loss import FocalLoss

from metrics.metrics import F1_Score, IoU_Score

config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='vaihingen_config.yaml', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

# Dataset parameters
group = parser.add_argument_group('Dataset parameters')
group.add_argument('--dataset', metavar='NAME', default='potsdam',
                   help='which kind of dataset')
# TRAIN + VAL
group.add_argument('--trainval-image-dir', metavar='NAME', default=r'D:/Potsdam_Cropped_1024_geoseg/train_val/image/',
                   help='path to image dataset')
group.add_argument('--trainval-label-dir', metavar='NAME', default=r'D:/Potsdam_Cropped_1024_geoseg/train_val/label/',
                   help='path to label dataset')
group.add_argument('--trainval-train-split', metavar='N', default=0.8, type=float,
                    help='dataset train split percentage (default: 0.8)')
group.add_argument('--trainval-val-split', metavar='N', default=0.2, type=float,
                    help='dataset validation split percentage (default: 0.2)')
# TEST
group.add_argument('--test-image-dir', metavar='NAME', default=r'D:\Potsdam_Cropped_1024_geoseg\test\image/',
                   help='path to image dataset')
group.add_argument('--test-label-dir', metavar='NAME', default=r'D:\Potsdam_Cropped_1024_geoseg\test\label/',
                   help='path to label dataset')
group.add_argument('--mosaic-ratio', type=float, default=0.25, metavar='M',
                    help='Mosaic Ratio')
# Model parameters
# Encoder
group = parser.add_argument_group('Encoder parameters')
group.add_argument('--encoder-model', default='swsl_resnet18', type=str, metavar='ENCODER_MODEL',
                   help='Name of model as encoder (default: "resnet34, swsl_resnet18"')
group.add_argument('--in-chans', type=int, default=3, metavar='N',  # in_chans
                   help='Image input channels (default: None => 3)')
group.add_argument('--pretrained', default=True, type=bool, metavar='ENCODER_MODEL_PRETRAINED',
                   help='Start encoder with pretrained version of specified network (if avail)')
# Reduction
group = parser.add_argument_group('Reduction parameters')
group.add_argument('--reduction-channels_scale', type=float, default=0.5,
                   metavar='REDUCTION_CHANNELS', help='channel reduction of output from encoder')
# Decoder
group = parser.add_argument_group('Decoder parameters')
group.add_argument('--depth', type=int, nargs='+', default=[1, 1, 1], metavar='N',
                    help='How many blocks run on each decoder ')
group.add_argument('--split-size', type=int, nargs='+', default=[8, 8, 8], metavar='N',
                    help='split size of each block')
group.add_argument('--num-heads', type=int, nargs='+', default=[8, 8, 8], metavar='N',
                    help='number of heads of each block')
# Decoder_Block
group.add_argument('--block-kernel-size', type=int, default=5, metavar='N',
                    help='kernel size on each block')

group.add_argument('--block-reduction', type=int, default=4, metavar='N',
                    help='reduction on each block')
group.add_argument('--block-attn-drop', type=float, default=0., metavar='N',
                    help='attention drop on each block')
group.add_argument('--block-qk-scale', type=float, default=None, metavar='N',
                    help='qk scale on each block')
# Decoder_Gating_Network
group.add_argument('--block-k', type=int, default=3, metavar='N',
                    help='selected expert on gating network')
group.add_argument('--block-hid-channels', type=int, default=512, metavar='N',
                    help='hidden channel on gating network pooling ')
group.add_argument('--block-pool-channels', type=int, default=16, metavar='N',
                    help='pool channel on gating network pooling')
group.add_argument('--block-pool-sizes', type=int, default=16, metavar='N',
                    help='pool size on gating network pooling')
# Decoder_Experts
group.add_argument('--block-num-experts', type=int, default=4, metavar='N',
                    help='amount of the experts')
group.add_argument('--block-loss-coef', type=int, default=0.1, metavar='N',
                    help='loss coefficients on gating network loss')
# Decoder_Channel_Attention_Experts
group.add_argument('--block-waves', type=str, nargs='+', default=['db1', 'sym2', 'coif1', 'bior1.3'], metavar='N',
                    help='channel attention experts')
# Decoder_Pixel_Attention_Experts
group.add_argument('--block-kernel-sizes', type=int, nargs='+', default=[1, 3, 7, 11], metavar='N',
                    help='pixel attention experts')
# Decoder_Feed_Forward_Network
group.add_argument('--block-drop-path', type=float, default=0.1, metavar='N',
                    help='drop path on feed forward network')
# Head
group.add_argument('--head-kernel-size', type=int, default=5, metavar='N',
                    help='kernel size on head')
group.add_argument('--num-classes', type=int, default=2, metavar='N',
                   help='number of label classes')
group.add_argument('--ignore-index', type=int, default=6, metavar='N',
                   help='ignore index')
# Optimizer parameters
group = parser.add_argument_group('Optimizer parameters')
group.add_argument('--opt', default='adamW', type=str, metavar='OPTIMIZER',
                   help='Optimizer (default: "sgd")')
group.add_argument('--momentum', type=float, default=0.9, metavar='M',
                   help='Optimizer momentum (default: 0.9)')
group.add_argument('--weight-decay', type=float, default=2e-5,
                   help='weight decay (default: 2e-5)')
# Learning rate schedule parameters
group = parser.add_argument_group('Learning rate schedule parameters')
group.add_argument('-b', '--batch-size', type=int, default=8, metavar='N',
                   help='Input batch size for training (default: 16)')
group.add_argument('--sched', type=str, default='cosine', metavar='SCHEDULER',
                   help='LR scheduler (default: "step"')
group.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                   help='learning rate, overrides lr-base if set (default: None)')
group.add_argument('--warmup-lr', type=float, default=1e-5, metavar='LR',
                   help='warmup learning rate (default: 1e-5)')
group.add_argument('--epochs', type=int, default=100, metavar='N',
                   help='number of epochs to train (default: 300)')
group.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                   help='epochs to warmup LR, if scheduler supports')
group.add_argument('--min-lr', type=float, default=1e-6, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (default: 0)')
# Augmentation & regularization parameters
# group = parser.add_argument_group('Augmentation and regularization parameters')
# group.add_argument('--no-aug', action='store_true', default=False,
#                    help='Disable all training augmentation, override other train aug args')
# Misc
group = parser.add_argument_group('Miscellaneous parameters')
group.add_argument('--device', type=str, default='cuda', metavar='DEVICE',
                   help='Choice on device (default : cuda:0)')
group.add_argument('--seed', type=int, default=69, metavar='S',
                   help='random seed (default: 42)')
group.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                   help='how many training processes to use (default: 4)')
group.add_argument('--prefetch', type=int, default=4, metavar='N',
                   help='how many batches prefetch (default: 4)')
group.add_argument('--output', metavar='NAME', default=r'D:/model_save/N/',
                   help='path to output folder (save model *.pth)')
group.add_argument('--checkpoint', metavar='TEST', default='IoU_Score.pth',
                   help='checkpoint (*.pth) for testing')
group.add_argument('--visualization-dir', metavar='NAME', default=r'D:/visualization/',
                   help='path for testing visualization')

# args = parser.parse_args()
# print(args)
def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text

def main():

    # config
    args, args_text = _parse_args()

    # seed
    seed_everything(args.seed)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    # tensorboard
    from tensorboardX import SummaryWriter
    mode = 'Resnet_encoder_decoder_wavelet_pixel_moe'
    writer = SummaryWriter(log_dir='log', filename_suffix=f'_{mode}')

    # device
    device = torch.device(args.device)

    # dataset
    # dataset
    if args.dataset == 'potsdam' or 'vaihingen':
        from datasets.pots_vai_dataset import SatelliteDataset
    elif args.dataset == 'loveda':
        from datasets.lov_dataset import SatelliteDataset
    else:
        from datasets.aer_dataset import SatelliteDataset

    # TEST
    dataset = SatelliteDataset(image_dir=args.test_image_dir,
                               label_dir=args.test_label_dir,
                               mode='test',
                               transform=None,
                               mosaic_ratio=args.mosaic_ratio)

    dataset_idx = np.arange(len(dataset))

    # TEST
    testdataset = Subset(dataset, dataset_idx)

    # dataloader
    testdataloader = DataLoader(testdataset,
                                 batch_size=args.batch_size,
                                 num_workers=args.workers,
                                 prefetch_factor=args.prefetch)

    # model
    from models.segmentation_model import Model
    model = Model(encoder_model=args.encoder_model,
                  in_channels=args.in_chans,
                  pretrained=args.pretrained,
                  reduction_channels_scale=args. reduction_channels_scale,
                  depth=args.depth,
                  classes=args.num_classes,
                  # block
                  block_kernel_size=args.block_kernel_size,
                  block_reduction=args.block_reduction,
                  block_attn_drop=args.block_attn_drop,
                  block_qk_scale=args.block_qk_scale,
                  # gating
                  block_k=args.block_k,
                  block_hid_channels=args.block_hid_channels,
                  block_pool_channels=args.block_pool_channels,
                  block_pool_sizes=args.block_pool_sizes,
                  # experts
                  block_num_experts=args.block_num_experts,
                  block_loss_coef=args.block_loss_coef,
                  # ca experts
                  block_waves=args. block_waves,
                  # pa experts
                  block_kernel_sizes=args.block_kernel_sizes,
                  # ffn
                  block_drop_path=args.block_drop_path,
                  # head
                  head_kernel_size=args.head_kernel_size,
                  )

    model.load_state_dict(torch.load(args.output + args.checkpoint))


    # loss functions #loss
    loss_fns = [CELoss(ignore_index=args.ignore_index),
                DiceLoss(ignore_index=args.ignore_index),
                FocalLoss(ignore_index=args.ignore_index)]

    # metric functions
    metric_fns = [F1_Score(), IoU_Score()]

    # visualization
    pred_path = args.visualization_dir

    # has aux loss
    test(device, model, testdataloader, loss_fns, metric_fns, pred_path, palette=args.dataset)


if __name__ == '__main__':
    main()
