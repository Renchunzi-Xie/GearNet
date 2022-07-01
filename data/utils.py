from data.transforms import transform
from data.transforms import test_transform
import numpy as np
import torch

def build_dataset(args, dataset, seed=1):
    if dataset == 'office31':
        from data.office31 import Office31
        args.num_classes = 31
        data_transform = transform(resize_size=256, crop_size=224)
        test1_transform = test_transform(resize_size=256, crop_size=224)
        source_dataset = Office31(domain=args.SourceDataset, transform=data_transform, args=args, seed=seed)
        target_dataset = Office31(domain=args.TargetDataset, transform=data_transform, args=args, seed=seed)
        test_dataset = Office31(domain=args.TargetDataset, transform=test1_transform, args=args, seed=seed)
    elif dataset == 'office_home':
        from data.office_home import OfficeHome
        args.num_classes = 65
        data_transform = transform(resize_size=256, crop_size=224)
        test1_transform = test_transform(resize_size=256, crop_size=224)
        source_dataset = OfficeHome(domain=args.SourceDataset, transform=data_transform, args=args, seed=seed)
        target_dataset = OfficeHome(domain=args.TargetDataset, transform=data_transform, args=args, seed=seed)
        test_dataset = OfficeHome(domain=args.TargetDataset, transform=test1_transform, args=args, seed=seed)

    return source_dataset, target_dataset, test_dataset

def generate_labels(net, data, args, device):
    net.eval()
    updated_labels = np.zeros(len(data))
    dataloader = torch.utils.data.DataLoader(
        data,
        batch_size=args.batch_size_target, shuffle=True,
        num_workers=4, pin_memory=True)
    with torch.no_grad():
        for batch_idx, (target_inputs, _, _, index) in enumerate(dataloader):
            try:
                class_output, _ = net(target_inputs.to(device))
            except:
                class_output, _, _ = net(target_inputs.to(device))
            pred = class_output.data.max(1, keepdim=True)[1]
            updated_labels[index] = pred[:,0].cpu().numpy()

    data.noisy_targets = updated_labels.tolist()
    dataloader = torch.utils.data.DataLoader(
        data,
        batch_size=args.batch_size_target, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True)
    return dataloader, data