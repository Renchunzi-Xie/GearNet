import torch
import torch.nn as nn
from models.feature_extractor.utils import build_models

class DAN_net(nn.Module):
    def __init__(self, args, bottleneck_dim):
        super(DAN_net, self).__init__()

        # Feature extractor and Classifier
        self.feature, self.feature_dim = build_models(args.arch)
        self.bottleneck_layer_list = [nn.Linear(self.feature_dim , bottleneck_dim),
                                      nn.BatchNorm1d(bottleneck_dim), nn.ReLU(), nn.Dropout(0.5)]
        self.bottleneck = nn.Sequential(*self.bottleneck_layer_list)
        self.classifier = nn.Linear(bottleneck_dim, args.num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        ## initialization
        self.bottleneck[0].weight.data.normal_(0, 0.005)
        self.bottleneck[0].bias.data.fill_(0.1)
        self.classifier.weight.data.normal_(0, 0.01)
        self.classifier.bias.data.fill_(0.0)

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.bottleneck(x)
        logits = self.classifier(x)
        softmax_outputs = self.softmax(logits)
        return x, logits, softmax_outputs