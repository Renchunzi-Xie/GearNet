from algs.gearnet_base_model_1 import GearNet_BaseModel
import torch.nn as nn
import torch

class GearNet_DANN(GearNet_BaseModel):
    def get_loss(self, source_inputs, source_labels, target_inputs):
        self.label_criterion = nn.CrossEntropyLoss()
        self.domain_criterion = nn.BCELoss()

        input_var = torch.cat((source_inputs, target_inputs), 0)
        y_var, y_softmax_var, d_var = self.net(input_var)
        source_y, target_y = y_var.chunk(2, 0)
        source_y_softmax, target_y_softmax = y_softmax_var.chunk(2, 0)
        source_d, target_d = d_var.chunk(2, 0)
        domain_label = self.generate_domain_labels()

        Ly = self.label_criterion(source_y, source_labels.long())

        # calculate Lkl
        Lkl = 0
        if self.args.step > 0:
            Lkl = self.cal_KL(target_inputs, target_y_softmax, reduce=True)
        # calculate Ld
        Ld = self.domain_criterion(d_var, domain_label)


        loss = Ly + 0.5 * Ld + 0.1 * Lkl
        self.Lc = Ly
        self.Ld = Ld
        return loss

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