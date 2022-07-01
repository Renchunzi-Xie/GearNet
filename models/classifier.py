import torch.nn as nn
from models.feature_extractor.utils import build_models

class Classifier(nn.Module):
    def __init__(self, args, bottleneck):
        super(Classifier, self).__init__()
        num_classes = args.num_classes
        # Feature extractor and Classifier
        self.feature, feature_dim = build_models(args.arch)
        self.bottleneck = nn.Linear(feature_dim, bottleneck)
        self.classifier = nn.Linear(bottleneck, num_classes)

        # initialize

        self.bottleneck.weight.data.normal_(0, 0.005)
        self.bottleneck.bias.data.fill_(0.1)
        self.classifier.weight.data.normal_(0, 0.01)
        self.classifier.bias.data.fill_(0.0)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.bottleneck(x)

        # classifier
        y = self.classifier(x)
        y_softmax = self.softmax(y)

        return y, y_softmax
