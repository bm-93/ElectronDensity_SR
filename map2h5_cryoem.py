import os
import random
import numpy as np
import gemmi
import pandas as pd
import h5py 
import argparse
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from mpl_toolkits.mplot3d import Axes3D
from functools import reduce
from scipy.spatial import distance
from scipy import ndimage
from scipy.ndimage.interpolation import rotate


data_dir = 'raw_data_cryoem'
map_files = [f for f in listdir(data_dir) if f.endswith('.map')]

patch_per_sample = 20
patch_size = 64
train_dir = 'data_cryoem_aug/train'
test_dir = 'data_cryoem_aug/test'

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

for fn in map_files:
    file_path = os.path.join(data_dir, fn)

    ccp4 = gemmi.read_ccp4_map(file_path)
    ccp4.setup()
    gt = np.array(ccp4.grid, copy=False)

    print(fn, gt.shape, gt.max())

    # filter out small protein structures
    if not np.all(np.array(gt.shape) > patch_size):
        continue

    for pn in range(patch_per_sample):

        start_x = random.randint(0, gt.shape[0]-patch_size)
        start_y = random.randint(0, gt.shape[1]-patch_size)
        start_z = random.randint(0, gt.shape[2]-patch_size)

        patch_gt = gt[start_x:start_x+patch_size, start_y:start_y+patch_size, start_z:start_z+patch_size]

        if pn > 4: 

            # random flip the patch
            flip_axes = []
            if bool(random.getrandbits(1)):
                flip_axes.append(0) 
            if bool(random.getrandbits(1)):
                flip_axes.append(1)         
            if bool(random.getrandbits(1)):
                flip_axes.append(2) 
            
            flip_axes = tuple(flip_axes)
            if len(flip_axes) > 0:
                patch_gt = np.flip(patch_gt, axis=flip_axes)
            
            # random rotate the patch
            max_angle = 30

            # rotate along x-axis
            angle = random.uniform(-max_angle, max_angle)
            patch_gt = rotate(patch_gt, angle, mode='nearest', axes=(0, 1), reshape=False)

            # rotate along y-axis
            angle = random.uniform(-max_angle, max_angle)
            patch_gt = rotate(patch_gt, angle, mode='nearest', axes=(0, 2), reshape=False)
            
            # rotate along z-axis
            angle = random.uniform(-max_angle, max_angle)
            patch_gt = rotate(patch_gt, angle, mode='nearest', axes=(1, 2), reshape=False)

        if random.random() < 0.2:
            save_dir = test_dir
        else:
            save_dir = train_dir

        # save the ground-truth patch to h5 file
        h5_fn = '{}_{}.h5'.format(fn.split('.')[0], pn)
        hf = h5py.File(os.path.join(save_dir, h5_fn), 'w')
        hf.create_dataset('data', data=patch_gt)
        hf.close()

        print('Patch {} has been saved!'.format(h5_fn))


# from utils import vtk_plot

# fn = os.path.join('raw_data_cryoem', 'emd_2512.map')

# ccp4 = gemmi.read_ccp4_map(fn)
# ccp4.setup()
# gt = np.array(ccp4.grid, copy=False)
# print(gt.shape)

# threshold = 0.0
# vtk_plot(gt, threshold, save_path='emd_2512.png')