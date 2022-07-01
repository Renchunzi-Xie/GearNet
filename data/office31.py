from PIL import Image
import os
import os.path
import numpy as np
import torch.utils.data as data
from data.noise_generater import generate_corrupted_labels
import copy

from random import choice

class Office31(data.Dataset):
    """
    load Office-31
    """
    def __init__(self, domain, transform, args, seed=1):
        self.train_imgs = []
        self.test_imgs = []
        np.random.seed(seed)

        self.transform = transform
        self.domain = domain
        data_path = "./datasets/office31/" + domain + "/images"
        walk(data_path, self.train_imgs)
        self.targets = [int(path.split("/")[-2]) for path in self.train_imgs]
        true_labels = copy.deepcopy(self.targets)
        self.noisy_targets = generate_corrupted_labels(true_labels, args)

    def __getitem__(self, index):
        img_path = self.train_imgs[index]
        target = self.targets[index]
        noisy_target = self.noisy_targets[index]
        image = Image.open(img_path).convert('RGB')
        img = self.transform(image)
        return img, target, noisy_target, index

    def __len__(self):
        return len(self.train_imgs)

def walk(path, file_list):
    ff = os.walk(path)
    for root, dirs, files in ff:
        for file in files:
            file_list.append(os.path.join(root, file))