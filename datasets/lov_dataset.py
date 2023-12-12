
"""
Dataset class for LoveDa

"""

from __future__ import annotations
from typing import Type, Callable, Tuple, Optional, Set, List, Union

import glob
from torch.utils.data import Dataset
import tifffile
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torchvision.transforms import ToTensor

import numpy as np
from PIL import Image


IMG_SIZE = (1024, 1024)


class SatelliteDataset(Dataset):
    def __init__(self,
                 image_dir,
                 label_dir,
                 mode: str = 'test',
                 transform=None,
                 mosaic_ratio: float = 0.25,
                 image_size: Tuple[int, int] = IMG_SIZE):

        """
        mode : trainval and test
        """
        self.image_size = image_size
        # prevent "default arguments value is mutable" warning when use `transform=[]`
        transform = [] if transform is None else transform

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

        self.mode = mode
        self.mosaic_ratio = mosaic_ratio
    def load_image_and_label(self, index):

        # load image and mask
        # image
        image = Image.open(self.image_list[index])
        image = np.array(image)
        # label(mask)

        label = Image.open(self.label_list[index])
        label = np.array(label)

        return image, label

    def load_mosaic_image_and_label(self, index):

        # create index list without current index
        index_list = np.append(np.arange(0, index), np.arange(index, self.len - 1))

        indices = np.append(index, np.random.choice(index_list, size=3))

        image_a, mask_a = self.load_image_and_label(indices[0])
        image_b, mask_b = self.load_image_and_label(indices[1])
        image_c, mask_c = self.load_image_and_label(indices[2])
        image_d, mask_d = self.load_image_and_label(indices[3])

        w, h = self.image_size

        # 1/4 to 3/4
        start_x, strat_y = w // 4, h // 4

        # The coordinates of the splice center
        offset_x, offset_y = np.random.randint(start_x, (w - start_x)), np.random.randint(strat_y, (h - strat_y))

        crop_size_a = (offset_x, offset_y)
        crop_size_b = (w - offset_x, offset_y)
        crop_size_c = (offset_x, h - offset_y)
        crop_size_d = (w - offset_x, h - offset_y)

        # transform
        random_crop_a = A.RandomCrop(width=crop_size_a[0], height=crop_size_a[1])
        random_crop_b = A.RandomCrop(width=crop_size_b[0], height=crop_size_b[1])
        random_crop_c = A.RandomCrop(width=crop_size_c[0], height=crop_size_c[1])
        random_crop_d = A.RandomCrop(width=crop_size_d[0], height=crop_size_d[1])

        croped_a = random_crop_a(image=image_a.copy(), mask=mask_a.copy())
        croped_b = random_crop_b(image=image_b.copy(), mask=mask_b.copy())
        croped_c = random_crop_c(image=image_c.copy(), mask=mask_c.copy())
        croped_d = random_crop_d(image=image_d.copy(), mask=mask_d.copy())

        img_crop_a, mask_crop_a = croped_a['image'], croped_a['mask']
        img_crop_b, mask_crop_b = croped_b['image'], croped_b['mask']
        img_crop_c, mask_crop_c = croped_c['image'], croped_c['mask']
        img_crop_d, mask_crop_d = croped_d['image'], croped_d['mask']

        top_image = np.concatenate((img_crop_a, img_crop_b), axis=1)
        bottom_image = np.concatenate((img_crop_c, img_crop_d), axis=1)
        image = np.concatenate((top_image, bottom_image), axis=0)
        image = np.ascontiguousarray(image)

        top_mask = np.concatenate((mask_crop_a, mask_crop_b), axis=1)
        bottom_mask = np.concatenate((mask_crop_c, mask_crop_d), axis=1)
        mask = np.concatenate((top_mask, bottom_mask), axis=0)
        mask = np.ascontiguousarray(mask)

        # img = Image.fromarray(img)
        # mask = Image.fromarray(mask)

        return image, mask

    def __getitem__(self, index):

        if np.random.random() > self.mosaic_ratio or self.mode == 'test':
            # no mosaic
            image, label = self.load_image_and_label(index)

        else:
            # mosaic
            image, label = self.load_mosaic_image_and_label(index)

        label[label == 0] = 255  # dont care
        label[label == 1] = 0
        label[label == 2] = 1
        label[label == 3] = 2
        label[label == 4] = 3
        label[label == 5] = 4
        label[label == 6] = 5
        label[label == 7] = 6

        # ToTensorV2
        if self.transform:
            transformed = self.transform(image=image, mask=label)
        else:
            transformed = ToTensorV2()(image=image, mask=label)


        image, label = transformed['image'], transformed['mask']
        # print(image.shape)

        self.count += 1

        return {'image': image,
                'label': label,
                'image_name': self.image_list[index],
                'label_name': self.label_list[index]}


    def __len__(self):

        return self.len
