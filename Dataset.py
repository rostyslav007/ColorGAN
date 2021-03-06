from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import numpy as np
from PIL import ImageCms
import os
import cv2 as cv


class ColorDataset(Dataset):
    '''
    Dataset for loading data
    '''
    def __init__(self, dir_path, img_size):
        self.dir_path = dir_path
        self.img_size = img_size
        self.paths = os.listdir(dir_path)[:1000]

    def __getitem__(self, idx):
        img_path = os.path.join(self.dir_path, self.paths[idx])
        img = Image.open(img_path).resize(self.img_size)
        x = img.convert('L')
        y = img
        transform_x = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5, ))
        ])
        transform_y = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        return transform_x(x), transform_y(y)

    def __len__(self):
        return len(self.paths)

