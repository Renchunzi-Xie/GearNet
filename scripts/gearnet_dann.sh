#!/usr/bin/env bash

for type in unif flip
do
  for level in 0.2 0.4
  do
    python gearnet_main.py --arch resnet50 --alg gearnet_dann --lr 0.003 --noise_level ${level}  --Dataset office31 --SourceDataset amazon --TargetDataset webcam --gpu 2 --epochs 200 --bottleneck 256 --batch_size_source 32  --batch_size_target 32 --noise_type ${type}
    python gearnet_main.py --arch resnet50 --alg gearnet_dann --lr 0.003 --noise_level ${level}  --Dataset office31 --SourceDataset amazon --TargetDataset dslr --gpu 2 --epochs 200 --bottleneck 256 --batch_size_source 32  --batch_size_target 32 --noise_type ${type}
    python gearnet_main.py --arch resnet50 --alg gearnet_dann --lr 0.003 --noise_level ${level}  --Dataset office31 --SourceDataset webcam --TargetDataset amazon --gpu 2 --epochs 200 --bottleneck 256 --batch_size_source 32  --batch_size_target 32 --noise_type ${type}
    python gearnet_main.py --arch resnet50 --alg gearnet_dann --lr 0.003 --noise_level ${level}  --Dataset office31 --SourceDataset webcam --TargetDataset dslr --gpu 2 --epochs 200 --bottleneck 256 --batch_size_source 32  --batch_size_target 32 --noise_type ${type}
    python gearnet_main.py --arch resnet50 --alg gearnet_dann --lr 0.003 --noise_level ${level}  --Dataset office31 --SourceDataset dslr --TargetDataset amazon --gpu 2 --epochs 200 --bottleneck 256 --batch_size_source 32  --batch_size_target 32 --noise_type ${type}
    python gearnet_main.py --arch resnet50 --alg gearnet_dann --lr 0.003 --noise_level ${level}  --Dataset office31 --SourceDataset dslr --TargetDataset webcam --gpu 2 --epochs 200 --bottleneck 256 --batch_size_source 32  --batch_size_target 32 --noise_type ${type}
  done
done