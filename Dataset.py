from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import numpy as np
import os
import cv2 as cv


class ColorDataset(Dataset):
    '''
    Dataset for loading data
    '''
    def __init__(self, dir_path, img_size):
        self.dir_path = dir_path
        self.img_size = img_size
        self.paths = os.listdir(dir_path)

    def __getitem__(self, idx):
        img_path = os.path.join(self.dir_path, self.paths[idx])
        lab_img = cv.imread(img_path, cv.COLOR_RGB2Lab)

        transf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        return {
            'lab': transf(lab_img)
        }

    def __len__(self):
        return min(len(self.paths), 500)

