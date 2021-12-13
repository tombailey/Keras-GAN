import os
from os.path import join

import torchvision.transforms as transforms
from torch.utils.data import Dataset

from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, folder_path):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))
        ])
        self.files = [
            join(folder_path, file) for file in os.listdir(folder_path)
        ]

    def __getitem__(self, index):
        return self.transform(Image.open(self.files[index % len(self.files)]))

    def __len__(self):
        return len(self.files)
