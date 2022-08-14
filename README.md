# ColorGAN

PyTorch Pix2PixGAN model implementation for image colorization task

#### Archive paper: https://arxiv.org/pdf/1611.07004.pdf
#### Pix2Pix paper with code: https://paperswithcode.com/method/pix2pix
#### U-Net model: https://arxiv.org/pdf/1505.04597.pdf
#### Patch GAN paper with code: https://paperswithcode.com/method/patchgan

## Project structure
**Dataset.py**: ColorDataset class for batch image loading

**Generator.py**: ***Generator U-NET*** like architecture 

**Discriminator.py**: ***Discriminator Patch-GAN*** like architecture

**config.py**: project configurations

**main.py**: project train loop

**train_utils.py**: helper functions for training process

**utils.py**: some aditional functions for data preprocessing

