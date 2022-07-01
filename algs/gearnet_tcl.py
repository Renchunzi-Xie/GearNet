from algs.gearnet_base_model_1 import GearNet_BaseModel
import torch.nn as nn
import torch
import math
import numpy as np

class GearNet_TCL(GearNet_BaseModel):
    def get_loss(self, source_inputs, source_labels, target_inputs):
        self.label_criterion = nn.CrossEntropyLoss()
        self.domain_criterion = nn.BCELoss()

        input_var = torch.cat((source_inputs, target_inputs), 0)
        y_var, y_softmax_var, d_var = self.net(input_var)
        source_y, target_y = y_var.chunk(2, 0)
        source_y_softmax, target_y_softmax = y_softmax_var.chunk(2, 0)
        source_d, target_d = d_var.chunk(2, 0)
        domain_label = self.generate_domain_labels()
        if self.iepoch < self.args.startiter:
            Ly = self.label_criterion(source_y, source_labels.long())
        else:
            Ly, source_weight, source_num = self.cal_Ly(source_y_softmax, source_d, source_labels, self.args)
            target_weight = torch.ones(source_weight.size()).to(self.device)

            # calculate Lt
        Lt = self.cal_Lt(target_y_softmax)

        # calculate Lkl
        Lkl = 0
        if self.args.step > 0:
            Lkl = self.cal_KL(target_inputs, target_y_softmax, reduce=True)

        # calculate Ld
        if self.iepoch < self.args.startiter:
            Ld = self.domain_criterion(d_var, domain_label)
        else:
            domain_weight = torch.cat([source_weight, target_weight], 0)
            domain_weight = domain_weight.view(-1, 1)
            domain_criterion = nn.BCELoss(weight=domain_weight).to(self.device)
            Ld = domain_criterion(d_var, domain_label)

        loss = Ly + Ld + 0.1 * Lt + 0.1 * Lkl
        self.Lc = Ly
        self.Ld = Ld
        return loss

    def cal_Lt(self, target_y_softmax):
        Gt_var = target_y_softmax
        Gt_en = - torch.sum((Gt_var * torch.log(Gt_var + 1e-8)), 1)
        Lt = torch.mean(Gt_en)
        return Lt

    def cal_Ly(self, source_y_softmax, source_d, label, args):
        label = label.long()
        agey = - math.log(args.Lythred)
        aged = - math.log(1.0 - args.Ldthred)
        age = agey + args.lambdad * aged
        y_softmax = source_y_softmax
        the_index = torch.LongTensor(np.array(range(args.batch_size_source))).to(self.device)
        y_label = y_softmax[the_index, label]
        y_loss = - torch.log(y_label)

        d_loss = - torch.log(1.0 - source_d)
        d_loss = d_loss.view(args.batch_size_source)

        weight_loss = y_loss + args.lambdad * d_loss

        weight_var = (weight_loss < age).float().detach()
        Ly = torch.mean(y_loss * weight_var)

        source_weight = weight_var.data.clone()
        source_num = float((torch.sum(source_weight)))
        return Ly, source_weight, source_num

    def set_optimizer(self):
        sgd_param = [
            {'params': self.net.bottleneck.parameters(), 'lr': 1},
            {'params': self.net.classifier.parameters(), 'lr': 1},
            {'params': self.net.dfc1.parameters(), 'lr': 1},
            {'params': self.net.dfc2.parameters(), 'lr': 1},
            {'params': self.net.discriminator.parameters(), 'lr': 1},
        ]
        self.optimizer = torch.optim.SGD(sgd_param, self.args.lr,
                                         momentum=self.args.momentum,
                                         weight_decay=self.args.decay)