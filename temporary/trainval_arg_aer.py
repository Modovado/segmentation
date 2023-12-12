from __future__ import annotations
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = 'True'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import argparse
import torch
from torch.utils.data import DataLoader
from dataset.aer_dataset import SatelliteDataset
from utils.utils import seed_everything, val, EpochMonitor
from utils.utils import aux_train

from losses.losses import CELoss, DiceLoss
from metrics.metrics import F1_Score, IoU_Score

from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler_v2, scheduler_kwargs

config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)

# Dataset parameters
group = parser.add_argument_group('Dataset parameters')

group.add_argument('--image-dir', metavar='NAME', default=r'D:/Potsdam_Cropped/all/image/',
                    help='path to image dataset')
group.add_argument('--label-dir', metavar='NAME', default=r'D:/Potsdam_Cropped/all/label/',
                    help='path to label dataset')

# TRAIN + VAL
group.add_argument('--train-image-dir', metavar='NAME', default=r'D:/AerialImageDataset/train/img_pieces/',
                   help='path to image dataset')
group.add_argument('--train-label-dir', metavar='NAME', default=r'D:/AerialImageDataset/train/label_pieces/',
                   help='path to label dataset')
group.add_argument('--val-image-dir', metavar='NAME', default=r'D:/AerialImageDataset/val/img_pieces/',
                   help='path to image dataset')
group.add_argument('--val-label-dir', metavar='NAME', default=r'D:/AerialImageDataset/val/label_pieces/',
                   help='path to label dataset')

# Model parameters
group = parser.add_argument_group('Model parameters')
group.add_argument('--encoder-model', default='swsl_resnet18', type=str, metavar='ENCODER_MODEL',
                    help='Name of model as encoder (default: "resnet34, swsl_resnet18"')
group.add_argument('--pretrained', default=True, type=bool, metavar='ENCODER_MODEL_PRETRAINED',
                   help='Start encoder with pretrained version of specified network (if avail)')
group.add_argument('--in-chans', type=int, default=3, metavar='N',  #in_chan
                    help='Image input channels (default: None => 3)')
group.add_argument('--num-classes', type=int, default=2, metavar='N',
                    help='number of label classes')
group.add_argument('-b', '--batch-size', type=int, default=8, metavar='N',  #batch
                    help='Input batch size for training (default: 16)')

# Optimizer parameters
group = parser.add_argument_group('Optimizer parameters')
group.add_argument('--opt', default='adamW', type=str, metavar='OPTIMIZER',  #opt adamW
                    help='Optimizer (default: "sgd")')
group.add_argument('--momentum', type=float, default=0.9, metavar='M',  #momentum
                    help='Optimizer momentum (default: 0.9)')
group.add_argument('--weight-decay', type=float, default=2e-5,  #weight 2e-5
                    help='weight decay (default: 2e-5)')

# Learning rate schedule parameters
group = parser.add_argument_group('Learning rate schedule parameters')
group.add_argument('--sched', type=str, default=None, metavar='SCHEDULER',  #scheduler 'cosine'
                    help='LR scheduler (default: "step"')
group.add_argument('--lr', type=float, default=1e-4, metavar='LR',  #lr
                    help='learning rate, overrides lr-base if set (default: None)')
group.add_argument('--warmup-lr', type=float, default=1e-5, metavar='LR',
                    help='warmup learning rate (default: 1e-5)')
group.add_argument('--epochs', type=int, default=100, metavar='N',  #epochs
                    help='number of epochs to train (default: 300)')
group.add_argument('--warmup-epochs', type=int, default=5, metavar='N',  #warmup
                    help='epochs to warmup LR, if scheduler supports')
group.add_argument('--min-lr', type=float, default=0, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (default: 0)')

# Augmentation & regularization parameters
group = parser.add_argument_group('Augmentation and regularization parameters')
group.add_argument('--no-aug', action='store_true', default=False,
                    help='Disable all training augmentation, override other train aug args')
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
group.add_argument('--checkpoint', metavar='TEST', default='PANET_best_score.pth',
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

    # tensorboard
    from tensorboardX import SummaryWriter
    mode = 'PANet_transblock_three_decoder'
    writer = SummaryWriter(log_dir='../log', filename_suffix=f'_{mode}')

    # device
    device = torch.device(args.device)

    # dataset
    # TRAIN + VAL
    traindataset = SatelliteDataset(image_dir=args.train_image_dir,
                                    label_dir=args.train_label_dir,
                                    transform=None,
                                    to_rgb=to_rgb)

    valdataset = SatelliteDataset(image_dir=args.val_image_dir,
                                  label_dir=args.val_label_dir,
                                  transform=None,
                                  to_rgb=to_rgb)

    # dataloader
    traindataloader = DataLoader(traindataset,
                                 batch_size=args.batch_size,
                                 shuffle=True,
                                 num_workers=args.workers,
                                 drop_last=True,
                                 prefetch_factor=args.prefetch)

    valdataloader = DataLoader(valdataset,
                               batch_size=args.batch_size,
                               num_workers=args.workers,
                               prefetch_factor=args.prefetch)


    # model
    from models.segmentation_model import Model
    model = Model(encoder_model=args.encoder_model,
                  pretrained=args.pretrained,
                  in_channels=args.in_chans,
                  classes=args.num_classes)

    # optimizer
    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))
    # print(optimizer)

    # scheduler
    updates_per_epoch = len(traindataloader)
    scheduler, num_epochs = create_scheduler_v2(
        optimizer,
        **scheduler_kwargs(args),
        updates_per_epoch=updates_per_epoch,
    )

    # loss functions #loss
    from losses.focal_loss import FocalLoss
    # loss_fns = [CELoss(ignore_index=5)]
    loss_fns = [CELoss(), DiceLoss(), FocalLoss()]
    # loss_fns = [CELoss(ignore_index=5, smooth=0.05), DiceLoss(ignore_index=5, smooth=0.05)]
    # loss_fns = [CELoss(ignore_index=5), FocalLoss(ignore_index=5)]

    # metric functions
    metric_fns = [F1_Score(), IoU_Score()]

    # EpochMonitor
    TrainMonitor = EpochMonitor()
    EvalMonitor = EpochMonitor()

    save_path = args.output

    num_epochs = args.epochs

    for epoch in range(num_epochs):

        # aux on self.training
        aux_train(device, epoch, model, traindataloader, optimizer, scheduler, loss_fns, metric_fns, TrainMonitor, writer)

        # has aux loss
        val(device, epoch, model, valdataloader, loss_fns, metric_fns, EvalMonitor, writer, save_path)

    writer.close()


if __name__ == '__main__':
    main()
