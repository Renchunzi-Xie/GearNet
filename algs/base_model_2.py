import torch
import sys
import torch.nn as nn
from algs.base_model_1 import BaseModel
from models.utils import build_alg_model

class BaseModel2(BaseModel):
    """
    This base model is for double models.
    """
    def __init__(self, source_loader, target_loader, test_loader, device, args):
        self.source = source_loader
        self.target = target_loader
        self.test_data = test_loader
        self.device = device
        self.args = args

        self.net = build_alg_model(args.alg, args, device)
        self.net2 = build_alg_model(args.alg, args, device)
        self.set_optimizer()

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
            accu_t, accu_t2 = self.test(self.test_data)
            print('\nAccuracy of the %s dataset: %f| %f\n' % (self.args.TargetDataset, accu_t, accu_t2))
            if accu_s > best_accu_s:
                best_accu_s = accu_s
            if accu_t > best_accu_t:
                best_accu_t = accu_t
        print('============ Summary ============= \n')
        print('Best accuracy of the %s dataset: %f' % (self.args.SourceDataset, best_accu_s))
        print('Best accuracy of the %s dataset: %f' % (self.args.TargetDataset, best_accu_t))


    def train_per_batch(self, data):
        self.args.global_iter = self.iepoch * self.len_source + (self.batch_idx + 1)
        self.adjust_learning_rate(self.optimizer1, self.args)
        self.adjust_learning_rate(self.optimizer2, self.args)

        source_inputs, _, source_labels, _ = data
        source_inputs_var = torch.autograd.Variable(source_inputs.to(self.device))
        source_labels_var = torch.autograd.Variable(source_labels.to(self.device))

        Ly1, Ly2 = self.get_loss(source_inputs_var, source_labels_var)

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

    def test(self, test_dataloader):
        self.net.eval()
        self.net2.eval()
        """ test """
        len_dataloader = len(test_dataloader)
        data_target_iter = iter(test_dataloader)

        i = 0
        n_total = 0
        n_correct = 0
        n_correct2 = 0
        while i < len_dataloader:
            t_img, t_label, _, _ = data_target_iter.__next__()
            batch_size = len(t_label)

            t_img = t_img.to(self.device)
            t_label = t_label.to(self.device)

            with torch.no_grad():
                _, class_output = self.net(t_img) #TODO: may change when the model output changes.
                pred = class_output.data.max(1, keepdim=True)[1] #TODO: other methods to do the prediction.
                n_correct += pred.eq(t_label.data.view_as(pred)).cpu().sum()
                n_total += batch_size

                _, class_output2 = self.net2(t_img)  # TODO: may change when the model output changes.
                pred2 = class_output2.data.max(1, keepdim=True)[1]  # TODO: other methods to do the prediction.
                n_correct2 += pred2.eq(t_label.data.view_as(pred)).cpu().sum()

                i += 1

            accu = n_correct.data.numpy() * 1.0 / n_total
            accu2 = n_correct2.data.numpy() * 1.0 / n_total
        return accu, accu2

    def print_loss(self):
        sys.stdout.write(
            '\r epoch: %d, [iter: %d / all %d], err_label1: %f, err_label2: %f' \
            % (self.iepoch, self.batch_idx + 1, len(self.source),
               self.Lc1.data.cpu().numpy(),
               self.Lc2.data.cpu().numpy()))
        sys.stdout.flush()

    def set_optimizer(self):
        sgd_param1 = [
            {'params': self.net.bottleneck.parameters(), 'lr': 1},
            {'params': self.net.classifier.parameters(), 'lr': 1},
        ]
        self.optimizer1 = torch.optim.SGD(sgd_param1, self.args.lr,
                                          momentum=self.args.momentum,
                                          weight_decay=self.args.decay)

        sgd_param2 = [
            {'params': self.net2.bottleneck.parameters(), 'lr': 1},
            {'params': self.net2.classifier.parameters(), 'lr': 1},
        ]
        self.optimizer2 = torch.optim.SGD(sgd_param2, self.args.lr,
                                          momentum=self.args.momentum,
                                          weight_decay=self.args.decay)

    def get_loss(self, source_inputs, source_labels):
        # TODO: Forwards & calculate loss
        loss1 = 0
        loss2 = 0
        return loss1, loss2

    def save_model(self):
        save_folder = self.args.save_path
        torch.save(self.net.state_dict(), save_folder + '/' + self.args.alg + 'net_1' + '.pt')
        torch.save(self.net2.state_dict(), save_folder + '/' + self.args.alg + 'net_2' + '.pt')

    def load_model(self, net1, net2):
        print('Load auxiliary model...')
        save_folder = self.args.save_path
        net1.load_state_dict(torch.load(save_folder + '/' + self.args.alg + 'net_1' + '.pt'))
        net2.load_state_dict(torch.load(save_folder + '/' + self.args.alg + 'net_2' + '.pt'))
        return net1, net2

