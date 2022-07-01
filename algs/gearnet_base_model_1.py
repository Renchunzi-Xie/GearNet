import torch
from models.utils import build_alg_model
from algs.base_model_1 import BaseModel

class GearNet_BaseModel(BaseModel):
    def __init__(self, source_loader, target_loader, test_loader, device, args):
        if args.direction == 0:
            self.source = source_loader
            self.target = target_loader
            self.test_data = test_loader
        else:
            self.source = target_loader
            self.target = source_loader
            self.test_data = source_loader
        self.args = args
        self.device = device
        self.net = build_alg_model(args.alg, args, device)
        self.set_optimizer()
        if args.step > 0:
            temp = args.SourceDataset
            args.SourceDataset = args.TargetDataset
            args.TargetDataset = temp
            self.aux_model = build_alg_model(args.alg, args, device)
            self.aux_model = self.load_model(self.aux_model)

    def train(self):
        self.net.train()
        best_accu_s = 0
        best_accu_t = 0
        self.len_target = len(self.target)
        self.len_source = len(self.source)
        for iepoch in range(self.args.epochs):
            self.iepoch = iepoch
            self.args.total_iter = self.args.epochs * len(self.source)
            for self.batch_idx, data in enumerate(self.source):
                self.args.global_iter = self.iepoch * self.len_source + (self.batch_idx + 1)
                self.train_per_batch(data)
                self.print_loss()
            accu_s = self.test(self.source)
            print('\nAccuracy of the %s dataset: %f' % (self.args.SourceDataset, accu_s))
            accu_t = self.test(self.test_data)
            print('\nAccuracy of the %s dataset: %f\n' % (self.args.TargetDataset, accu_t))
            if accu_s > best_accu_s:
                best_accu_s = accu_s
            if accu_t > best_accu_t:
                best_accu_t = accu_t
                self.save_model()
        print('============ Summary ============= \n')
        print('Best accuracy of the %s dataset: %f' % (self.args.SourceDataset, best_accu_s))
        print('Best accuracy of the %s dataset: %f' % (self.args.TargetDataset, best_accu_t))
        return best_accu_s, best_accu_t

    def cal_KL(self, source_inputs, soft_targets, reduce=True):
        with torch.no_grad():
            y_var, y_softmax_var, d_var = self.aux_model(source_inputs)
        kl_loss1 = self.kl_loss_compute(soft_targets, y_softmax_var, reduce=False)
        kl_loss2 = self.kl_loss_compute(y_softmax_var, soft_targets, reduce=False)
        if reduce:
            kl_loss = torch.mean(kl_loss1 + kl_loss2)
        else:
            kl_loss = kl_loss1 + kl_loss2
        return kl_loss

