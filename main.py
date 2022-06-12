import torch
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
from Dataset import ColorDataset
from train_utils import *
import torchvision
from utils import *
from PIL import ImageCms
from train_utils import get_crit_loss, get_gen_loss
import os
from torchvision.utils import make_grid
from tqdm import tqdm
import cv2 as cv
import numpy as np
from Generator import Generator
from Discriminator import Discriminator
from config import *


dataset = ColorDataset(dir_name, img_size)
data_loader = DataLoader(dataset, batch_size, shuffle=True)

generator = Generator(in_channels=1, out_channels=3, features=16).cuda()
discriminator = Discriminator(in_channels=4).cuda()

#x = torch.randn((2, 1, *img_size)).cuda()
#for x, y in data_loader:
#    grid = make_grid(y*0.5 + 0.5)
#    img = torchvision.transforms.ToPILImage()(grid)
#    img.show()
#exit()
gen_optim = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
disc_optim = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

bce = nn.BCEWithLogitsLoss()
L1 = nn.L1Loss()

disc_loss_list = []
gen_loss_list = []
d_scaler = torch.cuda.amp.GradScaler()
g_scaler = torch.cuda.amp.GradScaler()

#for x, y in data_loader:
#    pil_img = torchvision.transforms.ToPILImage()(y[0].clamp(-1, 1)*0.5 + 0.5)
#    grey = torchvision.transforms.ToPILImage()(x[0].clamp(-1, 1)*0.5 + 0.5)
#    plt.imshow(pil_img, cmap='gray')
#    plt.show()
#    plt.imshow(grey, cmap='gray')
#    plt.show()
#    break
#exit()


for e in range(num_epochs):
    for i, (x, y) in tqdm(enumerate(data_loader)):
        x, y = x.cuda(), y.cuda()
        fake_img = generator(x)

        with torch.cuda.amp.autocast():
            print(x.shape, fake_img.shape)
            disc_preds_fake = discriminator(x, fake_img.detach())
            disc_preds_real = discriminator(x, y)

            disc_fake_loss = bce(disc_preds_fake, torch.zeros_like(disc_preds_fake))
            disc_real_loss = bce(disc_preds_real, torch.ones_like(disc_preds_real))
            disc_loss = disc_real_loss + disc_fake_loss

        disc_optim.zero_grad()
        d_scaler.scale(disc_loss).backward()
        d_scaler.step(disc_optim)
        d_scaler.update()

        disc_loss_list.append(disc_loss.item())

        #learn generator
        with torch.cuda.amp.autocast():
            disc_preds = discriminator(x, fake_img.detach())
            gen_loss = -bce(disc_preds, torch.zeros_like(disc_preds)) + 100*L1(fake_img, y)

        gen_optim.zero_grad()
        g_scaler.scale(gen_loss).backward()
        g_scaler.step(gen_optim)
        g_scaler.update()

        gen_loss_list.append(gen_loss.item())

        print('\ndisc loss: ', disc_loss_list[-1], ' gen loss: ', gen_loss_list[-1])
        if e % 20 == 0 and i == 0:
            imgs = fake_img
            print(torch.abs(fake_img - y).max())
            grid = make_grid((imgs * 0.5 + 0.5).cpu().detach())
            img = torchvision.transforms.ToPILImage()(grid)
            img.show()

    if e % 50 == 0:
        disc_optim = torch.optim.Adam(discriminator.parameters(), lr=0.0002 - (e // 50) * 0.00001, betas=(0.5, 0.999))
        gen_path = os.path.join('models', 'generator', 'generator_epoch_' + str(e) + '.pt')
        disc_path = os.path.join('models', 'discriminator', 'discriminator_epoch_' + str(e) + '.pt')
        torch.save(generator.state_dict(), gen_path)
        torch.save(discriminator.state_dict(), disc_path)


generator = Generator(in_channels=1, out_channels=3)
generator.load_state_dict(torch.load('models/generator/generator_epoch_20.pt'))

for x, y in data_loader:
    imgs = generator(x)*0.5 + 0.5
    grid = make_grid(imgs.cpu().detach())
    img = torchvision.transforms.ToPILImage()(grid)
    img.show()
    break









