
"""
Dataset class for Aerial

train   : img_pieces / label_pieces
val     : img_pieces / label_pieces
test    : img_pieces / label_pieces

Credit : https://github.com/KyanChen/STT/blob/main/Tools/CutImgSegWithLabel.py
"""
import glob
from torch.utils.data import Dataset
import cv2
import tifffile
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torchvision.transforms import ToTensor

import numpy as np

class SatelliteDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, to_rgb=False):

        # prevent "default arguments value is mutable" warning when use `transform=[]`
        transform = [] if transform is None else transform

        # img_pieces / label_pieces
        self.image_dir = image_dir
        self.label_dir = label_dir

        # [absolute path]
        self.image_list = sorted(glob.glob(self.image_dir + '*.png'), key=len)
        self.label_list = sorted(glob.glob(self.label_dir + '*.png'), key=len)

        # test
        # self.image_list = self.image_list[0:100]
        # self.label_list = self.label_list[0:100]

        # assert when len no matching
        assert len(self.image_list) == len(self.label_list), \
            f' img_list len `{len(self.image_list)}` is not equal to mask_list len `{len(self.label_list)}` '

        # list-form
        self.transform = transform

        self.len = len(self.image_list)

        self.count = 0
        self.to_rgb = to_rgb
    def __getitem__(self, index):

        # loaded data
        image = cv2.imread(self.image_list[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype('int16')


        if self.to_rgb:
            image = image[:, :, :3]

        label = cv2.imread(self.label_list[index])
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY).astype('int16')

        label[label > 0] = 1

        # ToTensorV2
        if self.transform:
            transformed = self.transform(image=image, mask=label)
        else:
            transformed = ToTensorV2()(image=image, mask=label)

        image, label = transformed['image'], transformed['mask']

        self.count += 1

        return {'image': image,
                'label': label,
                'image_name': self.image_list[index],
                'label_name': self.label_list[index]}

    def __len__(self):

        return self.len