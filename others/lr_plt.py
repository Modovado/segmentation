from __future__ import annotations
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = 'True'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import argparse
import numpy as np
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split
from dataset.pots_vai_dataset import SatelliteDataset

from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler_v2, scheduler_kwargs

config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)

# Dataset parameters
group = parser.add_argument_group('Dataset parameters')


# [TRAIN + VAL]
group.add_argument('--trainval-image-dir', metavar='NAME', default=r'D:/Potsdam_Cropped_1024_geoseg/train_val/image/', # geoseg
                   help='path to image dataset')
group.add_argument('--trainval-label-dir', metavar='NAME', default=r'D:/Potsdam_Cropped_1024_geoseg/train_val/label/',
                   help='path to label dataset')
group.add_argument('--trainval-train-split', metavar='N', default=0.8, type=float,
                    help='dataset train split percentage (default: 0.8)')
group.add_argument('--trainval-val-split', metavar='N', default=0.2, type=float,
                    help='dataset validation split percentage (default: 0.2)')

# Model parameters
group = parser.add_argument_group('Model parameters')
group.add_argument('--encoder-model', default='swsl_resnet18', type=str, metavar='ENCODER_MODEL',
                    help='Name of model as encoder (default: "resnet34, swsl_resnet18"')
group.add_argument('--pretrained', default=True, type=bool, metavar='ENCODER_MODEL_PRETRAINED',
                   help='Start encoder with pretrained version of specified network (if avail)')
group.add_argument('--in-chans', type=int, default=3, metavar='N',  #in_chan  #geoseg use 3 chans
                    help='Image input channels (default: None => 3)')
group.add_argument('--num-classes', type=int, default=5, metavar='N', #geoseg # try 7 classes(6 + bg)
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
group.add_argument('--sched', type=str, default='cosine', metavar='SCHEDULER',  #scheduler 'cosine'
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


from matplotlib import pyplot as plt

def get_lr_per_epoch(scheduler, num_epoch):
    lr_per_epoch = []
    for epoch in range(num_epoch):
        # lr_per_epoch.append(scheduler.get_epoch_values(epoch))
        lr_per_epoch.append(scheduler._get_values(epoch))
    return lr_per_epoch



def main():
    # dataset
    # TRAIN + VAL
    dataset = SatelliteDataset(image_dir=args.trainval_image_dir,
                               label_dir=args.trainval_label_dir,
                               transform=None)


    dataset_idx = np.arange(len(dataset))

    # TRAIN + VAL
    traindataset_idx, valdataset_idx = train_test_split(dataset_idx,
                                                        train_size=args.trainval_train_split,
                                                        test_size=args.trainval_val_split,
                                                        random_state=args.seed)

    traindataset, valdataset = Subset(dataset, traindataset_idx), Subset(dataset, valdataset_idx)

    # dataloader
    traindataloader = DataLoader(traindataset,
                                 batch_size=args.batch_size,
                                 shuffle=True,
                                 num_workers=args.workers,
                                 drop_last=True,
                                 prefetch_factor=args.prefetch)

    # model
    from models.decoder_four_ import PANet
    # from models.res18_baseline import PANet
    model = PANet(encoder_model=args.encoder_model,
                  pretrained=args.pretrained,
                  in_channels=args.in_chans,
                  classes=args.num_classes, )  #### size might be problematic

    # optimizer
    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))


    # scheduler
    updates_per_epoch = len(traindataloader)
    scheduler, num_epochs = create_scheduler_v2(
        optimizer,
        **scheduler_kwargs(args),
        updates_per_epoch=updates_per_epoch,
    )
    # [PYTORCH_SCHED]
    num_epochs = args.epochs



    lr_per_epoch = get_lr_per_epoch(scheduler, num_epochs)

    plt.plot([i for i in range(num_epochs)], lr_per_epoch);


    print(scheduler.state_dict())
    plt.show()


if __name__ == '__main__':
    main()
