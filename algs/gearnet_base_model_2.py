import torch
from models.utils import build_alg_model
from algs.base_model_2 import BaseModel2

class GearNet_BaseModel2(BaseModel2):
    def __init__(self, source_loader, target_loader, test_loader, device, args):
        if args.direction == 0:
            self.source = source_loader
            self.target = target_loader
            self.test_data = test_loader
        else:
            self.source = target_loader
            self.target = source_loader
            self.test_data = source_loader
        self.device = device
        self.args = args
        self.net = build_alg_model(args.alg, args, device)
        if args.step > 0:
            temp = args.SourceDataset
            args.SourceDataset = args.TargetDataset
            args.TargetDataset = temp
            self.aux_model1 = build_alg_model(args.alg, args, device)
            self.aux_model2 = build_alg_model(args.alg, args, device)
            self.aux_model1, self.aux_model2 = self.load_model(self.aux_model1, self.aux_model2)

        sgd_param1 = [
            {'params': self.net.bottleneck.parameters(), 'lr': 1},
            {'params': self.net.classifier.parameters(), 'lr': 1},
        ]
        self.optimizer1 = torch.optim.SGD(sgd_param1, args.lr,
                                         momentum=args.momentum, weight_decay=args.decay)
        self.net2 = build_alg_model(args.alg, args, device)
        sgd_param2 = [
            {'params': self.net2.bottleneck.parameters(), 'lr': 1},
            {'params': self.net2.classifier.parameters(), 'lr': 1},
        ]
        self.optimizer2 = torch.optim.SGD(sgd_param2, args.lr,
                                          momentum=args.momentum, weight_decay=args.decay)

    def train(self):
        self.net.train()
        self.net2.train()
        best_accu_s = 0
        best_accu_t = 0
        self.len_target = len(self.target)
        self.len_source = len(self.source)

        for iepoch in range(self.args.epochs):
            self.iepoch = iepoch
            self.args.total_iter = self.args.epochs * len(self.source)
            for self.batch_idx, data in enumerate(self.source):
                self.train_per_batch(data)
                self.print_loss()
            accu_s, accu_s2 = self.test(self.source)
            print('\nAccuracy of the %s dataset: %f| %f' % (self.args.SourceDataset, accu_s, accu_s2))
            accu_t, accu_t2 = self.test(self.target)
            print('\nAccuracy of the %s dataset: %f| %f\n' % (self.args.TargetDataset, accu_t, accu_t2))
            if accu_s > best_accu_s:
                best_accu_s = accu_s
            if accu_t > best_accu_t:
                best_accu_t = accu_t
                self.save_model()
        print('============ Summary ============= \n')
        print('Best accuracy of the %s dataset: %f' % (self.args.SourceDataset, best_accu_s))
        print('Best accuracy of the %s dataset: %f' % (self.args.TargetDataset, best_accu_t))
        return best_accu_s, best_accu_t

    def train_per_batch(self, data):
        self.args.global_iter = self.iepoch * self.len_source + (self.batch_idx + 1)
        self.adjust_learning_rate(self.optimizer1, self.args)
        self.adjust_learning_rate(self.optimizer2, self.args)

        source_inputs, _, source_labels, _ = data
        source_inputs_var = torch.autograd.Variable(source_inputs.to(self.device))
        source_labels_var = torch.autograd.Variable(source_labels.to(self.device))
        target_inputs, _, _, _ = self.sample_target(self.batch_idx, self.len_target)
        target_inputs_var = torch.autograd.Variable(target_inputs.to(self.device))

        Ly1, Ly2 = self.get_loss(source_inputs_var, source_labels_var, target_inputs_var)

        # backwards
        self.optimizer1.zero_grad()
        self.net.zero_grad()
        Ly1.backward()
        self.optimizer1.step()

        self.optimizer2.zero_grad()
        self.net2.zero_grad()
        Ly2.backward()
        self.optimizer2.step()

        self.Lc1 = Ly1
        self.Lc2 = Ly2

    def get_loss(self, source_inputs, source_labels, target_inputs):
        # TODO: Forwards & calculate loss
        loss1 = 0
        loss2 = 0
        return loss1, loss2

    def cal_KL(self, source_inputs, soft_targets1, soft_targets2, reduce=True):
        with torch.no_grad():
            y_var1, y_softmax_var1 = self.aux_model1(source_inputs)
            y_var2, y_softmax_var2 = self.aux_model2(source_inputs)
        kl_loss1 = self.kl_loss_compute(soft_targets1, y_softmax_var1, reduce=False)
        kl_loss2 = self.kl_loss_compute(y_softmax_var1, soft_targets1, reduce=False)
        kl_loss3 = self.kl_loss_compute(soft_targets2, y_softmax_var2, reduce=False)
        kl_loss4 = self.kl_loss_compute(y_softmax_var2, soft_targets2, reduce=False)
        if reduce:
            kl_loss_1 = torch.mean(kl_loss1 + kl_loss2)
            kl_loss_2 = torch.mean(kl_loss3 + kl_loss4)
        else:
            kl_loss_1 = kl_loss1 + kl_loss2
            kl_loss_2 = kl_loss3 + kl_loss4
        return kl_loss_1, kl_loss_2
