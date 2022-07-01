import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
#
# class ResNet(nn.Module):
#     model = models.__dict__[args.arch](pretrained=True)
#     model = nn.Sequential(*list(model.children())[:-1])
    


class ResNet(nn.Module):
    def __init__(self, arch):
        super(ResNet, self).__init__()
        # models
        model = models.__dict__[arch](pretrained=True)
        self.feature_dim = model.fc.in_features
        self.model = nn.Sequential(*list(model.children())[:-1])


    def forward(self, x):
        out = self.model(x)
        return out

    def get_feature_dim(self):
        """
        :return: The dimension of output features.
        """
        return self.feature_dim


def ResNet18():
    return ResNet('resnet18')

def ResNet34():
    return ResNet('resnet34')

def ResNet50():
    return ResNet('resnet50')

def ResNet101():
    return ResNet('resnet101')

def ResNet152():
    return ResNet('resnet152')

