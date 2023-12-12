from __future__ import annotations
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = 'True'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import argparse
import numpy as np
import torch
from torch.utils.data import Subset, DataLoader

from dataset.pots_vai_dataset import SatelliteDataset
from utils.utils import seed_everything, test
from losses.losses import CELoss, DiceLoss
from metrics.metrics import F1_Score, IoU_Score

config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)

# Dataset parameters

group = parser.add_argument_group('Dataset parameters')

# TEST
group.add_argument('--test-image-dir', metavar='NAME', default=r'D:\Potsdam_Cropped_1024_geoseg\test\image/', #_geoseg
                   help='path to image dataset')
group.add_argument('--test-label-dir', metavar='NAME', default=r'D:\Potsdam_Cropped_1024_geoseg\test\label/',
                   help='path to label dataset')

# Model parameters
group = parser.add_argument_group('Model parameters')
group.add_argument('--encoder-model', default='swsl_resnet18', type=str, metavar='ENCODER_MODEL',
                    help='Name of model as encoder (default: "resnet34, swsl_resnet18"')
group.add_argument('--pretrained', default=True, type=bool, metavar='ENCODER_MODEL_PRETRAINED',
                   help='Start encoder with pretrained version of specified network (if avail)')
group.add_argument('--in-chans', type=int, default=3, metavar='N',  #in_chans #geoseg use 3 chans
                   help='Image input channels (default: None => 3)')
group.add_argument('--num-classes', type=int, default=6, metavar='N', #geoseg
                   help='number of label classes')
group.add_argument('-b', '--batch-size', type=int, default=8, metavar='N',  #batch
                   help='Input batch size for training (default: 16)')

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
group.add_argument('--checkpoint', metavar='TEST', default='IoU_Score.pth',  #checkpoint

                   help='checkpoint (*.pth) for testing')
group.add_argument('--visualization-dir', metavar='NAME', default=r'D:/visualization/',
                   help='path for testing visualization')

args = parser.parse_args()
# print(args)


def main():

    # if in_chans is 3, then it's training on rgb data
    to_rgb = True if args.in_chans == 3 else False

    # seed
    seed_everything(args.seed)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    # device
    device = torch.device(args.device)

    # dataset
    # TEST
    dataset = SatelliteDataset(image_dir=args.test_image_dir,
                               label_dir=args.test_label_dir,
                               mode='test',
                               transform=None,
                               mosaic_ratio=0.25)

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
    # from models.res18_baseline import PANet
    model = Model(encoder_model=args.encoder_model,
                  pretrained=args.pretrained,
                  in_channels=args.in_chans,
                  classes=args.num_classes)

    model.load_state_dict(torch.load(args.output + args.checkpoint))

    # loss functions #loss

    # 5 is unfavorable class,
    # (geoseg) 6 is fully black cropped

    from losses.focal_loss import FocalLoss
    loss_fns = [CELoss(ignore_index=6), DiceLoss(ignore_index=6), FocalLoss(ignore_index=6)]
    # loss_fns = [CELoss(ignore_index=5), DiceLoss(ignore_index=5)]
    # loss_fns = [CELoss(ignore_index=5)]

    # metric functions
    metric_fns = [F1_Score(), IoU_Score()]

    # visualization
    pred_path = args.visualization_dir

    # has aux loss
    test(device, model, testdataloader, loss_fns, metric_fns, pred_path)


if __name__ == '__main__':
    main()
