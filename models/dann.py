import torch
import torch.nn as nn
from models.feature_extractor.utils import build_models
import math

class GRL_Layer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gamma, global_iter, total_iter):
        ctx.gamma = gamma
        ctx.global_iter = global_iter
        ctx.total_iter = total_iter
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx, grad_output):
        lamda = 2.0 / (1.0 + math.exp(- ctx.gamma * ctx.global_iter / ctx.total_iter)) - 1
        return (- lamda) * grad_output, None, None, None

class DANN_net(nn.Module):
    def __init__(self, args, bottleneck, num_hidden):
        super(DANN_net, self).__init__()
        num_classes = args.num_classes
        # Feature extractor and Classifier
        self.args = args
        self.feature, feature_dim = build_models(args.arch)
        self.bottleneck = nn.Linear(feature_dim, bottleneck)
        self.classifier = nn.Linear(args.bottleneck, num_classes)

        # Domain Discriminator
        self.relu = nn.ReLU()
        self.dfc1 = nn.Linear(bottleneck, num_hidden)
        self.dfc2 = nn.Linear(num_hidden, num_hidden)
        self.discriminator = nn.Linear(num_hidden, 1)
        self.sigmoid = nn.Sigmoid()
        self.drop1 = nn.Dropout(0.5)
        self.drop2 = nn.Dropout(0.5)

        # initialize
        self.bottleneck.weight.data.normal_(0, 0.005)
        self.bottleneck.bias.data.fill_(0.1)
        self.classifier.weight.data.normal_(0, 0.01)
        self.classifier.bias.data.fill_(0.0)
        self.dfc1.weight.data.normal_(0, 0.01)
        self.dfc1.bias.data.fill_(0.0)
        self.dfc2.weight.data.normal_(0, 0.01)
        self.dfc2.bias.data.fill_(0.0)
        self.discriminator.weight.data.normal_(0, 0.3)
        self.discriminator.bias.data.fill_(0.0)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.bottleneck(x)

        # classifier
        y = self.classifier(x)
        y_softmax = self.softmax(y)

        # domain discriminator
        xd = GRL_Layer.apply(x, self.args.gamma, self.args.global_iter, self.args.total_iter)

        # xd with grl
        xd = self.drop1(self.relu(self.dfc1(xd)))
        xd = self.drop2(self.relu(self.dfc2(xd)))
        d = self.discriminator(xd)
        d = self.sigmoid(d)
        return y, y_softmax, d



