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

from models import *
from dataset import EMDataset
from dice_loss import diceloss


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
    parser.add_argument("--glr", type=float, default=0.0002, help="adam: generator learning rate")
    parser.add_argument("--dlr", type=float, default=0.0002, help="adam: discriminator learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
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
    parser.add_argument("--data_dir", type=str, default='data', help="dataset directory")
    parser.add_argument("--gpu", type=str, default='cpu', help="gpu id or cpu")
    opt = parser.parse_args()
    print(opt)

    start_time = time.time()
    save_name = strftime("%m%d%H%M", gmtime())
    save_name += '_vox2vox'

    os.makedirs("validations/%s" % save_name, exist_ok=True)
    os.makedirs("saved_models/%s" % save_name, exist_ok=True)

    # cuda = True if torch.cuda.is_available() else False
    # device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    cuda = False if opt.gpu == 'cpu' else True
    device = torch.device(opt.gpu)
    if cuda:
        torch.cuda.set_device(device)

    # Loss functions
    criterion_GAN = torch.nn.MSELoss()
    criterion_voxelwise = diceloss()

    # Loss weight of L1 voxel-wise loss between translated image and real image
    lambda_voxel = 5

    # Calculate output of image discriminator (PatchGAN)
    patch = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4, opt.img_depth // 2 ** 4)

    # Initialize generator and discriminator
    generator = GeneratorUNet()
    discriminator = ConditionalDiscriminator()

    if cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        criterion_GAN.cuda()
        criterion_voxelwise.cuda()
        # generator = generator.to(device)
        # discriminator = discriminator.to(device)
        # criterion_GAN.to(device)
        # criterion_voxelwise.to(device)


    if opt.epoch != 0:
        # Load pretrained models
        generator.load_state_dict(torch.load("saved_models/%s/generator_%d.pth" % (save_name, opt.epoch)))
        discriminator.load_state_dict(torch.load("saved_models/%s/discriminator_%d.pth" % (save_name, opt.epoch)))
    else:
        # Initialize weights
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.glr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.dlr, betas=(opt.b1, opt.b2))

    # Configure dataloaders
    transforms_ = transforms.Compose([
        # transforms.Resize((opt.img_height, opt.img_width, opt.img_depth), Image.BICUBIC),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataloader = DataLoader(
        EMDataset(opt.data_dir + "/train", transforms_=transforms_),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
    )

    val_dataloader = DataLoader(
        EMDataset(opt.data_dir + "/test", transforms_=transforms_),
        batch_size=1,
        shuffle=True,
        num_workers=1,
    )

    # Tensor type
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    def sample_voxel_volumes(epoch):
        """Saves a generated sample from the validation set"""
        imgs = next(iter(val_dataloader))
        real_A = Variable(imgs["A"].unsqueeze_(1).type(Tensor))
        real_B = Variable(imgs["B"].unsqueeze_(1).type(Tensor))
        fake_B = generator(real_A)

        # convert to numpy arrays
        real_A = real_A.cpu().detach().numpy()
        real_B = real_B.cpu().detach().numpy()
        fake_B = fake_B.cpu().detach().numpy()

        image_folder = "validations/%s/epoch_%s_" % (save_name, epoch)

        hf = h5py.File(image_folder + 'real_A.vox', 'w')
        hf.create_dataset('data', data=real_A)

        hf1 = h5py.File(image_folder + 'real_B.vox', 'w')
        hf1.create_dataset('data', data=real_B)

        hf2 = h5py.File(image_folder + 'fake_B.vox', 'w')
        hf2.create_dataset('data', data=fake_B)

    # ----------
    #  Training
    # ----------

    prev_time = time.time()
    discriminator_update = 'False'
    for epoch in range(opt.epoch, opt.n_epochs):
        for i, batch in enumerate(dataloader):

            # Model inputs
            real_A = Variable(batch["A"].unsqueeze_(1).type(Tensor))
            real_B = Variable(batch["B"].unsqueeze_(1).type(Tensor))

            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)

            # ---------------------
            #  Train Discriminator, only update every disc_update batches
            # ---------------------
            # Real loss
            fake_B = generator(real_A)
            pred_real = discriminator(real_B, real_A)
            loss_real = criterion_GAN(pred_real, valid)

            # Fake loss
            pred_fake = discriminator(fake_B.detach(), real_A)
            loss_fake = criterion_GAN(pred_fake, fake)
            # Total loss
            loss_D = 0.5 * (loss_real + loss_fake)

            d_real_acu = torch.ge(pred_real.squeeze(), 0.5).float()
            d_fake_acu = torch.le(pred_fake.squeeze(), 0.5).float()
            d_total_acu = torch.mean(torch.cat((d_real_acu, d_fake_acu), 0))

            if d_total_acu <= opt.d_threshold:
                optimizer_D.zero_grad()
                loss_D.backward()
                optimizer_D.step()
                discriminator_update = 'True'

            # ------------------
            #  Train Generators
            # ------------------
            optimizer_D.zero_grad()
            optimizer_G.zero_grad()

            # GAN loss
            fake_B = generator(real_A)
            pred_fake = discriminator(fake_B, real_A)
            loss_GAN = criterion_GAN(pred_fake, valid)
            # Voxel-wise loss
            loss_voxel = criterion_voxelwise(fake_B, real_B)

            # Total loss
            loss_G = loss_GAN + lambda_voxel * loss_voxel

            loss_G.backward()

            optimizer_G.step()

            batches_done = epoch * len(dataloader) + i

            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f, D accuracy: %f, D update: %s] [G loss: %f, voxel: %f, adv: %f] ETA: %s"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    loss_D.item(),
                    d_total_acu,
                    discriminator_update,
                    loss_G.item(),
                    loss_voxel.item(),
                    loss_GAN.item(),
                    time_left,
                )
            )
            # If at sample interval save image
            if batches_done % (opt.sample_interval*len(dataloader)) == 0:
                sample_voxel_volumes(epoch)
                print('*****volumes sampled*****')

            discriminator_update = 'False'

        if opt.checkpoint_interval != -1 and (epoch+1) % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(generator.state_dict(), "saved_models/%s/generator_%d.pth" % (save_name, epoch))
            torch.save(discriminator.state_dict(), "saved_models/%s/discriminator_%d.pth" % (save_name, epoch))

    # save the model after training 
    torch.save(generator.state_dict(), "saved_models/%s/generator.pth" % (save_name))
    torch.save(discriminator.state_dict(), "saved_models/%s/discriminator.pth" % (save_name))


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    train()
