import argparse
import torch
import numpy as np
from data.utils import build_dataset
from algs.utils import create_alg
from data.utils import generate_labels

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='alexnet')
parser.add_argument('--alg', default='standard', type=str,
                    help='algorithm')
parser.add_argument('--Dataset', default='office31', type=str,
                    help='dataset')
parser.add_argument('--SourceDataset', default='amazon', type=str,
                    help='source dataset')
parser.add_argument('--TargetDataset', default='webcam', type=str,
                    help='target dataset')
parser.add_argument('-bc', '--bottleneck', default=256, type=int, metavar='N',
                    help='width of bottleneck (default: 256)')
parser.add_argument('-hl', '--num_hidden', default=1024, type=int, metavar='N',
                    help='width of hiddenlayer (default: 1024)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--gpu', type=str, default=None)
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate.')
parser.add_argument('--batch_size_source', type=int, default=100, help='Batch Size of source domain.')
parser.add_argument('--batch_size_target', type=int, default=100, help='Batch Size of target domain.')
parser.add_argument('--epochs', type=int, default=50, help='The number of epochs.')
parser.add_argument('--num_classes', type=int, default=31, help='The number of classes.')
parser.add_argument('--noise_type', default='unif', type=str, metavar='M', help='The type of noise.')
parser.add_argument('--noise_level', default=0, type=float, help='Noise level')
parser.add_argument('--alpha', default=10.0, type=float, metavar='M') #adjust learning rate
parser.add_argument('--beta', default=0.75, type=float, metavar='M') #adjust learning rate
parser.add_argument('--gamma', default=10.0, type=float, metavar='M',
                    help='dloss weight')#GRL layer
parser.add_argument('--global_iter', default=10.0, type=float, metavar='M')#GRL layer
parser.add_argument('--total_iter', default=10.0, type=float, metavar='M')#GRL layer
# tcl
parser.add_argument('--startiter', default=30, type=int)
parser.add_argument('--Lythred', default=0.5, type=float)
parser.add_argument('--Ldthred', default=0.5, type=float)
parser.add_argument('--lambdad',default=1.0,type=float)
# gearnet
parser.add_argument('--direction', default=0, type=int)
parser.add_argument('--step', default=0, type=int)
parser.add_argument('--save_path', default='./save_model', type=str)

args = parser.parse_args()
# Set seeds
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# GUP
import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
if args.gpu is not None:
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

# load data
source_data, target_data, test_data = build_dataset(args, args.Dataset, seed=1)
print('loading source dataset...')
source_loader = torch.utils.data.DataLoader(
                source_data,
                batch_size=args.batch_size_source, shuffle=True,
                num_workers=4, pin_memory=True, drop_last=True)
print('loading target dataset...')
target_loader = torch.utils.data.DataLoader(
                target_data,
                batch_size=args.batch_size_target, shuffle=True,
                num_workers=4, pin_memory=True, drop_last=True)
print('loading test dataset...')
test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=args.batch_size_target, shuffle=False,
            num_workers=4, pin_memory=True, drop_last=True)

#Define model
acc = []
for args.step in range(10):
    args.direction = args.step % 2
    alg_obj = create_alg(source_loader, target_loader, test_loader, device, args)
    step_accs, step_acct = alg_obj.train()
    if args.direction == 0:
        target_loader, _ = generate_labels(alg_obj.net, target_data, args, device)
        acc.append(step_acct)
    else:
        acc.append(step_accs)
print("\nThe backbone method result is %f" %(acc[0]))
print("\nGearNet method result is %f" %(max(acc[1:])))


