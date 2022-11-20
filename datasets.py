import torch
import torch.utils.data
import torchvision
from glob import glob
import os
import cv2
import numpy as np

def by_sample_number(path):
    num = int(path.split(os.path.sep)[-1].split('.')[0])
    return num


class MNISTDataset(torchvision.datasets.MNIST):
    """
    ### MNIST dataset
    """

    def __init__(self, image_size, data_path):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(image_size),
            torchvision.transforms.ToTensor(),
        ])
        self.data_path = data_path
        super().__init__(self.data_path, train=True, download=True, transform=transform)

    def __getitem__(self, item):
        return super().__getitem__(item)[0]

class CarsDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path):
        self.imsize = 224
        self.dataset_path = dataset_path
        self._load_dataset_paths()
    
    def _load_dataset_paths(self):
        self.all_samples = sorted(glob(self.dataset_path + os.path.sep + '*.jpg'), key=by_sample_number)
    
    def __getitem__(self, index):
        img = cv2.imread(self.all_samples[index])
        img = cv2.resize(img, (self.imsize, self.imsize))
        img = img.transpose(2, 0, 1)
        assert img.shape == (3, self.imsize, self.imsize)
        assert np.max(img) <= 255
        img = torch.FloatTensor(img / 255.)
        return img
    
    def __len__(self):
        return len(self.all_samples)
