import torch
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
from Dataset import ColorDataset
from train_utils import *
import torchvision
from utils import *
import os
from torchvision.utils import make_grid
from tqdm import tqdm
import cv2 as cv
import numpy as np
from Generator import Generator
from Discriminator import Discriminator
from config import *
torch.backends.cudnn.benchmark = True


dataset = ColorDataset(dir_name, img_size)
data_loader = DataLoader(dataset, batch_size, shuffle=True)

generator = Generator(in_list, out_list, img_size).cuda()
discriminator = Discriminator(in_channels=3).cuda()

#x = torch.randn((2, 1, *img_size)).cuda()

#print(generator(x).shape)

#exit()
generator_loss = GeneratorLoss()
discriminator_loss = DiscriminatorLoss()
gen_optim = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
disc_optim = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

disc_loss_list = []
gen_loss_list = []
scaler = torch.cuda.amp.GradScaler()

for e in range(num_epochs):
    for i, data in tqdm(enumerate(data_loader)):
        real_images = data['lab'].cuda()

        with torch.cuda.amp.autocast():
            l_channel = real_images[:, 0:1, :, :]
            ab_channels = real_images[:, 1:, :, :]
            fake_ab = generator(l_channel)

            disc_preds_fake = discriminator(l_channel, fake_ab.detach())
            disc_preds_real = discriminator(l_channel, ab_channels)

            disc_fake_loss = discriminator_loss(disc_preds_fake, torch.zeros_like(disc_preds_fake))
            disc_real_loss = discriminator_loss(disc_preds_real, torch.ones_like(disc_preds_real))
            disc_loss = (disc_real_loss + disc_fake_loss) / 2

        disc_optim.zero_grad()
        scaler.scale(disc_loss).backward()
        scaler.step(disc_optim)
        scaler.update()

        disc_loss_list.append(disc_loss.item())

        #learn Generator
        with torch.cuda.amp.autocast():
            generated_ab = generator(l_channel)
            disc_preds = discriminator(l_channel, generated_ab.detach())
            gen_loss = generator_loss(torch.ones_like(disc_preds), disc_preds, generated_ab, ab_channels)

        gen_optim.zero_grad()
        scaler.scale(gen_loss).backward()
        scaler.step(gen_optim)
        scaler.update()

        gen_loss_list.append(gen_loss.item())

        print('\ndisc loss: ', disc_loss_list[-1], ' gen loss: ', gen_loss_list[-1])

    if e % 10 == 0:
        gen_path = os.path.join('models', 'generator', 'generator_epoch_' + str(e) + '.pt')
        disc_path = os.path.join('models', 'discriminator', 'discriminator_epoch_' + str(e) + '.pt')
        torch.save(generator.state_dict(), gen_path)
        torch.save(discriminator.state_dict(), disc_path)

generator = Generator(in_list, out_list, img_size)
generator.load_state_dict(torch.load('models/generator/generator_epoch_10.pt'))

for data in data_loader:
    gan_batch = data['lab']
    lab_c = gan_batch[:, 0:1, :, :]
    ab_channels = generator(lab_c)
    img_tensor = torch.cat([lab_c, ab_channels], dim=1)
    print(img_tensor.shape)
    grid = make_grid(img_tensor, normalize=True)
    img = torchvision.transforms.ToPILImage()(grid)
    plt.imshow(img)
    plt.show()
    lab_tensor2cv_rgb(img_tensor[0], 'test.png')
    break









