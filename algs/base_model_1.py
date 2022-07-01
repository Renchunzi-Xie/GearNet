import torch
import sys
from models.utils import build_alg_model
import torch.nn.functional as F

class BaseModel():
    """
    This base model is for single model.
    """
    def __init__(self, source_loader, target_loader, test_loader, device, args):
        self.source = source_loader
        self.target = target_loader
        self.test_data = test_loader
        self.device = device
        self.args = args

        self.net = build_alg_model(args.alg, args, device)
        self.set_optimizer()

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

    def train_per_batch(self, data):
        self.args.global_iter = self.iepoch * self.len_source + (self.batch_idx + 1)
        self.adjust_learning_rate(self.optimizer, self.args)
        target_inputs, _, _, _ = self.sample_target(self.batch_idx, self.len_target)
        source_inputs, _, source_labels, _ = data
        source_inputs = torch.autograd.Variable(source_inputs.to(self.device))
        source_labels = torch.autograd.Variable(source_labels.to(self.device))
        target_inputs = torch.autograd.Variable(target_inputs.to(self.device))
        loss = self.get_loss(source_inputs, source_labels, target_inputs)

    #     backwards
        self.optimizer.zero_grad()
        self.net.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_loss(self, source_inputs, source_labels, target_inputs):
        # TODO: Forwards & calculate loss
        loss = 0
        return loss

    def print_loss(self):
        sys.stdout.write(
            '\r epoch: %d, [iter: %d / all %d], err_label: %f, err_domain: %f' \
            % (self.iepoch, self.batch_idx + 1, len(self.source),
               self.Lc.data.cpu().numpy(),
               self.Ld.data.cpu().numpy()))
        sys.stdout.flush()

    def test(self, test_dataloader):
        self.net.eval()
        """ test """
        len_dataloader = len(test_dataloader)
        data_target_iter = iter(test_dataloader)

        i = 0
        n_total = 0
        n_correct = 0

        while i < len_dataloader:
            t_img, t_label, _, _ = data_target_iter.__next__()
            batch_size = len(t_label)

            t_img = t_img.to(self.device)
            t_label = t_label.to(self.device)

            with torch.no_grad():
                try:
                    _, class_output, _ = self.net(t_img) #TODO: may change when the model output changes.
                except:
                    _, class_output = self.net(t_img) #TODO: may change when the model output changes.
                pred = class_output.data.max(1, keepdim=True)[1] #TODO: other methods to do the prediction.
                n_correct += pred.eq(t_label.data.view_as(pred)).cpu().sum()
                n_total += batch_size

                i += 1

            accu = n_correct.data.numpy() * 1.0 / n_total
        return accu


    def adjust_learning_rate(self, optimizer, args):
        lr = args.lr * (1.0 + args.alpha * args.global_iter / args.total_iter) ** (-args.beta)
        for i, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr
        return lr

    def sample_target(self, step, n_batches):
        global iter_target
        if step % n_batches == 0:
            iter_target = iter(self.target)
        return iter_target.__next__()

    def sample_source(self, step, n_batches):
        global iter_source
        if step % n_batches == 0:
            iter_source = iter(self.source)
        return iter_source.__next__()

    def set_optimizer(self):
        sgd_param = [
            {'params': self.net.bottleneck.parameters(), 'lr': 1},
            {'params': self.net.classifier.parameters(), 'lr': 1},
        ]
        self.optimizer = torch.optim.SGD(sgd_param, self.args.lr,
                                         momentum=self.args.momentum, weight_decay=self.args.decay)

    def generate_domain_labels(self):
        source_domain_label = torch.FloatTensor(self.args.batch_size_source, 1)
        target_domain_label = torch.FloatTensor(self.args.batch_size_target, 1)
        source_domain_label.fill_(1)
        target_domain_label.fill_(0)
        domain_label = torch.cat([source_domain_label.to(self.device), target_domain_label.to(self.device)], 0)
        return domain_label

    def kl_loss_compute(self, pred, soft_targets, reduce=True):

        kl = F.kl_div(torch.log(pred), soft_targets, reduction='none')

        if reduce:
            return torch.mean(torch.sum(kl, dim=1))
        else:
            return torch.sum(kl, 1)

    def save_model(self):
        save_folder = self.args.save_path
        torch.save(self.net.state_dict(), save_folder + '/' + self.args.alg + '.pt')

    def load_model(self, net):
        print('Load auxiliary model...')
        save_folder = self.args.save_path
        net.load_state_dict(torch.load( save_folder + '/' + self.args.alg + '.pt'))
        return net
