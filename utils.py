import vtk
from vtk.util import numpy_support
import h5py
import numpy as np
import pyvista as pv
import random
import time
import datetime
import sys
import torch
from torch.autograd import Variable
import torch.nn.functional as F


def vtk_plot(f, threshold=None, save_path=None):
    # # binary thresholding of intensity range [-1 1]
    if threshold != None:
        f[f <= threshold] = -1.
        f[f > threshold] = 1.

    # plot 3d segmentation mask with pyvista
    mesh = pv.wrap(f)
    plotter = pv.Plotter()
    plotter.add_mesh_threshold(mesh, cmap='PuBuGn', smooth_shading=True, lighting=True)
    if save_path != None:
        cpos = plotter.show(screenshot=save_path, use_panel=False)
    else:
        cpos = plotter.show()
    plotter.close()


class ReplayBuffer:
    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)
