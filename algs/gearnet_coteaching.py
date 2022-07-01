from algs.gearnet_base_model_2 import GearNet_BaseModel2
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class GearNet_CoTeaching(GearNet_BaseModel2):
    def get_loss(self, source_inputs, source_labels, target_inputs):
        y1, _ = self.net(source_inputs)
        y2, _ = self.net2(source_inputs)

        if self.iepoch < 10:
            forget_rate = 0
        else:
            forget_rate = self.args.noise_level
        loss1, loss2 = self.loss_coteaching(y1, y2, source_labels, forget_rate)
        if self.args.step > 0:
            _, soft_target1 = self.net(target_inputs)
            _, soft_target2 = self.net2(target_inputs)
            kl_loss1, kl_loss2 = self.cal_KL(target_inputs, soft_target1, soft_target2)
            loss1 += 0.1 * kl_loss1
            loss2 += 0.1 * kl_loss2

        self.Lc1 = loss1
        self.Lc2 = loss2
        return loss1, loss2

    def loss_coteaching(self, y_1, y_2, t, forget_rate):
        t = t.long()
        label_criterion = nn.CrossEntropyLoss(reduction='none')
        loss_1 = label_criterion(y_1, t).cpu()
        ind_1_sorted = np.argsort(loss_1.data)
        loss_1_sort = loss_1[ind_1_sorted]

        loss_2 = label_criterion(y_2, t).cpu()
        ind_2_sorted = np.argsort(loss_2.data)
        loss_2_sort = loss_2[ind_2_sorted]

        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * len(loss_1_sort))

        ind_1_update = ind_1_sorted[:num_remember]
        ind_2_update = ind_2_sorted[:num_remember]

        if len(ind_1_update) == 0:
            ind_1_update = ind_1_sorted.cpu().numpy()
            ind_2_update = ind_2_sorted.cpu().numpy()

        # exchange
        loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
        loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])
        return loss_1_update, loss_2_update



