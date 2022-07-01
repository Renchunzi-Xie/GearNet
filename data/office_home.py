from PIL import Image
import numpy as np
import torch.utils.data as data
from data.noise_generater import generate_corrupted_labels
import torchvision.datasets as dset
import copy

class OfficeHome(data.Dataset):
    def __init__(self, domain, transform, args, seed=1):
        np.random.seed(seed)
        self.transform = transform
        self.domain = domain
        data_path = "./datasets/OfficeHomeDataset_10072016/" + domain
        self.datasets = dset.ImageFolder(data_path, transform)
        self.targets = self.datasets.targets
        true_labels = copy.deepcopy(self.targets)
        self.noisy_targets = generate_corrupted_labels(true_labels, args)

    def __getitem__(self, index):
        target = self.targets[index]
        noisy_target = self.noisy_targets[index]
        img = self.datasets[index][0]
        return img, target, noisy_target, index

    def __len__(self):
        return len(self.datasets)
