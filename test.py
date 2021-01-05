import argparse
import os
import numpy as np
import math
import itertools
import datetime
import sys
import h5py
import torch
import time

from time import gmtime, strftime
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import directed_hausdorff

from models import *
from dataset import EMDataset
from dice_loss import diceloss


def test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_height", type=int, default=64, help="size of image height")
    parser.add_argument("--img_width", type=int, default=64, help="size of image width")
    parser.add_argument("--img_depth", type=int, default=64, help="size of image depth")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--disc_update", type=int, default=5, help="only update discriminator every n iter")
    parser.add_argument("--d_threshold", type=int, default=.8, help="discriminator threshold")
    parser.add_argument("--threshold", type=int, default=-1, help="threshold during sampling, -1: No thresholding")
    parser.add_argument("--sample_interval", type=int, default=20, help="interval between sampling of images from generators")
    parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")
    parser.add_argument("--model_dir", type=str, default="07191728", help="directory to load models")
    parser.add_argument("--save_dir", type=str, default="test", help="directory to save test results")
    opt = parser.parse_args()
    print(opt)

    image_folder = os.path.join(opt.save_dir, opt.model_dir)
    os.makedirs(image_folder, exist_ok=True)

    cuda = True if torch.cuda.is_available() else False
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    # Loss functions
    criterion_GAN = torch.nn.MSELoss()
    criterion_voxelwise = diceloss()

    # Loss weight of L1 voxel-wise loss between translated image and real image
    lambda_voxel = 100

    # Calculate output of image discriminator (PatchGAN)
    patch = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4, opt.img_depth // 2 ** 4)

    # Initialize generator and discriminator
    generator = GeneratorUNet()
    # discriminator = Discriminator()

    if cuda:
        generator = generator.cuda()
        # discriminator = discriminator.cuda()
        # criterion_GAN.cuda()
        criterion_voxelwise.cuda()

    # load models
    generator.load_state_dict(torch.load("saved_models/%s/generator.pth" % (opt.model_dir), map_location=device))

    # Configure dataloaders
    transforms_ = transforms.Compose([
        transforms.ToTensor(),
    ])

    val_dataloader = DataLoader(
        EMDataset("data/test", transforms_=transforms_),
        batch_size=1,
        shuffle=True,
        num_workers=1,
    )

    # Tensor type
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # ----------
    #  Testing
    # ----------

    min_val = -1.5
    max_val = 2.5
    l1_error_pred, l1_error_input = [], []
    l2_error_pred, l2_error_input = [], []
    hausdorff_pred, hausdorff_input = [], []

    for i, batch in enumerate(val_dataloader):
        info = batch["info"]
        input = Variable(batch["A"].unsqueeze_(1).type(Tensor))
        gt = Variable(batch["B"].unsqueeze_(1).type(Tensor))
        pred = generator(input)

        dice_input = criterion_voxelwise(pred, gt)
        dice_pred = criterion_voxelwise(input, gt)

        print(dice_input.item(), dice_pred.item())

        # convert to numpy arrays
        pred = pred.cpu().detach().numpy()
        pred = (pred * 0.5 + 0.5) * (max_val - min_val) + min_val
        input = input.cpu().detach().numpy()
        input = (input * 0.5 + 0.5) * (max_val - min_val) + min_val
        gt = gt.cpu().detach().numpy()
        gt = (gt * 0.5 + 0.5) * (max_val - min_val) + min_val

        l1_error_input.append(np.mean(np.abs((gt-input))))
        l1_error_pred.append(np.mean(np.abs((gt-pred))))
        l2_error_input.append(np.mean(np.square((gt-input))))
        l2_error_pred.append(np.mean(np.square((gt-pred))))

        save_fn = info[0].split('/')[-1] + '.h5'
        save_path = os.path.join(image_folder, save_fn)
        print(save_path)
        hf = h5py.File(save_path, 'w')
        hf.create_dataset('data', data=pred)
        hf.close()

        save_fn = info[0].split('/')[-1] + '.npy'
        save_path = os.path.join(image_folder+'_npy', save_fn)
        np.save(save_path, pred)

        print("{} saved!".format(save_path))

    l1_error_input = np.array(l1_error_input)
    l1_error_pred = np.array(l1_error_pred)
    l2_error_input = np.array(l2_error_input)
    l2_error_pred = np.array(l2_error_pred)

    print("l1 error input:", np.mean(l1_error_input))
    print("l1 error pred:", np.mean(l1_error_pred))
    print("l2 error input:", np.mean(l2_error_input))
    print("l2 error pred:", np.mean(l2_error_pred))

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    test()
