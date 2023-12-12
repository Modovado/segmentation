
"""
Dataset class for Potsdam / Vaihingen

"""
import glob
from torch.utils.data import Dataset
import tifffile
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torchvision.transforms import ToTensor

import numpy as np
from PIL import Image


#
# # image_path = r'D:\Potsdam_Cropped_1024_geoseg\train_val\image\top_potsdam_2_10_0_0.tif'
# image_path = r'C:\Users\User\PycharmProjects\GeoSeg-main\data\potsdam\train_images\top_potsdam_2_10_RGB.tif'
#
# # loaded data
# image = tifffile.imread(image_path).astype('int16') # for tiff
#
# image = np.array(image)
#
#
# print(image.shape)

# import random
# index = 10
# end = 100
#
# index_list = np.append(np.arange(0, index), np.arange(index,  end - 1))
#
# indices = np.append(index, np.random.choice(index_list, size=3))


# print(indices)

import random
# w, h = 1024, 1024
#
# start_x, strat_y = w // 4, h // 4
# # 1/4 to 3/4
# # The coordinates of the splice center
# offset_x, offset_y = np.random.randint(start_x, (w - start_x)), np.random.randint(strat_y, (h - strat_y))
#
#
# crop_size_a = (offset_x, offset_y)
# crop_size_b = (w - offset_x, offset_y)
# crop_size_c = (offset_x, h - offset_y)
# crop_size_d = (w - offset_x, h - offset_y)
#
# print(crop_size_a)
# print(crop_size_b)
# print(crop_size_c)
# print(crop_size_d)

print(np.random.random())