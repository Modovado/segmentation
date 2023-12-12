

"""
Dataset class for Potsdam / Vaihingen

"""
import glob

# pots
# label_dir = f'D:/Potsdam_Cropped_1024_geoseg/test\image/'

# vai
# label_dir = f'D:\Vaihingen_Cropped_1024_geoseg/test\image/'

# lov
label_dir = r'D:\LoveDa_Cropped\test\all\image/'

# label_list = sorted(glob.glob(label_dir + '*.tif'), key=len)
label_list = sorted(glob.glob(label_dir + '*.png'), key=len)
print(label_list[1127])
