import os
os.environ["CUDA_LAUNCH_BLOCKING"] = 'True'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import glob
from torch.utils.data import Dataset, DataLoader, Subset
import tifffile
import argparse
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler_v2, scheduler_kwargs


config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
# Dataset parameters
group = parser.add_argument_group('Dataset parameters')

group.add_argument('--image-dir', metavar='NAME', default=r'D:/Potsdam_Cropped/all/image/',
                   help='path to image dataset')
group.add_argument('--label-dir', metavar='NAME', default=r'D:/Potsdam_Cropped/all/label/',
                   help='path to label dataset')
group.add_argument('--train-split', metavar='N', default=0.7, type=float,
                   help='dataset train split percentage (default: 0.7)')
group.add_argument('--val-split', metavar='N', default=0.1, type=float,
                   help='dataset validation split percentage (default: 0.1)')
group.add_argument('--test-split', metavar='N', default=0.2, type=float,
                   help='dataset test split percentage (default: 0.2)')

# Model parameters
group = parser.add_argument_group('Model parameters')
group.add_argument('--encoder-model', default='resnet34', type=str, metavar='ENCODER_MODEL',
                   help='Name of model as encoder (default: "resnet34"')
group.add_argument('--pretrained', default=True, type=bool, metavar='ENCODER_MODEL_PRETRAINED',
                   help='Start encoder with pretrained version of specified network (if avail)')
group.add_argument('--in-chans', type=int, default=4, metavar='N',
                   help='Image input channels (default: None => 3)')
group.add_argument('--num-classes', type=int, default=5, metavar='N',
                   help='number of label classes')
group.add_argument('-b', '--batch-size', type=int, default=16, metavar='N',
                   help='Input batch size for training (default: 16)')

# Optimizer parameters
group = parser.add_argument_group('Optimizer parameters')
group.add_argument('--opt', default='adam', type=str, metavar='OPTIMIZER',
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
group = parser.add_argument_group('Augmentation and regularization parameters')
group.add_argument('--no-aug', action='store_true', default=False,
                   help='Disable all training augmentation, override other train aug args')
# Misc
group = parser.add_argument_group('Miscellaneous parameters')
group.add_argument('--device', type=str, default='cuda', metavar='DEVICE',
                   help='Choice on device (default : cuda:0)')
group.add_argument('--seed', type=int, default=42, metavar='S',
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

def get_lr_per_epoch(scheduler, num_epoch):
    lr_per_epoch = []
    for epoch in range(num_epoch):
        lr_per_epoch.append(scheduler.get_epoch_values(epoch))
    return lr_per_epoch

if __name__ == '__main__':


    # model
    from models.new_baseline_model import UNet
    model = UNet(encoder_model=args.encoder_model,
                 pretrained=args.pretrained,
                 in_channels=args.in_chans,
                 classes=args.num_classes)
    # optimizer
    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))
    # print(optimizer)

    # scheduler
    scheduler, num_epochs = create_scheduler_v2(
        optimizer,
        **scheduler_kwargs(args))
    from matplotlib import pyplot as plt
    lr_per_epoch = get_lr_per_epoch(scheduler, args.epochs)

    plt.plot([i for i in range(args.epochs)], lr_per_epoch)
    plt.show()






     # image_dir = r'F:\Vaihingen\top/'
    # image_list = sorted(glob.glob(image_dir + '*.tif'), key=len)
    # min_idx, max_idx = 0, 0
    # min_shape, max_shape = (0, 0), (0, 0)
    # min, max = float('inf'), 0
    #
    # for image_idx, image_path in enumerate(image_list):
    #     image = tifffile.imread(image_path).astype('int16')
    #     h, w, c = image.shape
    #
    #     if h * w > max:
    #         max_idx = image_idx
    #         max_shape = (h, w)
    #         max = h * w
    #
    #     if h * w < min:
    #         min_idx = image_idx
    #         min_shape = (h, w)
    #         min = h * w
    #
    # print(f'{max_idx=}, {max_shape=}, {max=}')
    # print(f'{min_idx=}, {min_shape=}, {min=}')


