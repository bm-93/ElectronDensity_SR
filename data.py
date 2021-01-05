import os
import random
import numpy as np
import gemmi
import pandas as pd
import h5py 
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from mpl_toolkits.mplot3d import Axes3D
from functools import reduce
from scipy.spatial import distance
from scipy import ndimage


data_dir = 'data'
data_files = []
for path, subdirs, files in os.walk(data_dir):
    for name in files:
        if name.endswith('_input.h5'):
            data_files.append(os.path.join(path, name))

max_val = 0.0
min_val = 0.0
for fn in data_files:
    image = h5py.File(fn).get('data')[()]
    max_val = max((np.max(image), max_val))
    min_val = min((np.min(image), min_val))

    print(np.sum(np.where(image > 1)), np.sum(np.where(image > 2)), np.max(image))

print(min_val, max_val)