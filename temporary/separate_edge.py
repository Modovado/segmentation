
"""
Dataset class for Aerial

train   : img_pieces / label_pieces
val     : img_pieces / label_pieces
test    : img_pieces / label_pieces

Credit : https://github.com/KyanChen/STT/blob/main/Tools/CutImgSegWithLabel.py
"""
import glob
import shutil

from torch.utils.data import Dataset
import cv2
import tifffile
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torchvision.transforms import ToTensor

import numpy as np
from pathlib import Path

# img_pieces / label_pieces
image_dir = r'D:\Potsdam_Cropped_1024_geoseg\test\image/'
label_dir = r'D:\Potsdam_Cropped_1024_geoseg\test\label/'

# img_pieces / label_pieces have no black-edged
image_no_black_edge_dir = r'D:\Potsdam_Cropped_1024_geoseg\test\image_nb/'
label_no_black_edge_dir = r'D:\Potsdam_Cropped_1024_geoseg\test\label_nb/'

# img_pieces / label_pieces have black-edged
image_black_edge_dir = r'D:\Potsdam_Cropped_1024_geoseg\test\image_b/'
label_black_edge_dir = r'D:\Potsdam_Cropped_1024_geoseg\test\label_b/'

# [absolute path]
image_list = sorted(glob.glob(image_dir+'*.tif'), key=len)
label_list = sorted(glob.glob(label_dir + '*.png'), key=len)

for idx, (image, label) in enumerate(zip(image_list, label_list)):

    # image name
    image_path = Path(image)
    label_path = Path(label)

    # loaded data
    image = cv2.imread(str(image_path))

    # height and width
    H, W, _ = image.shape

    black_px = 0
    total_px = H * W

    for i in range(H):
        for j in range(W):
            if image[i, j].sum() == 0:
                black_px +=1

    black_ratio = black_px / total_px
    print(black_ratio)

    # black edged
    if black_ratio >= 0.140625:
        shutil.copyfile(image_path, f'{image_black_edge_dir}/{image_path.name}')
        shutil.copyfile(label_path, f'{label_black_edge_dir}/{label_path.name}')

    # not black edged
    else:
        shutil.copyfile(image_path, f'{image_no_black_edge_dir}/{image_path.name}')
        shutil.copyfile(label_path, f'{label_no_black_edge_dir}/{label_path.name}')
