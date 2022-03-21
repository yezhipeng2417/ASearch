import os

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

class Trainer:
    def __init__(self, args, model, train_data, dev_data, loss_func, optimizer, device=None, rank=0):
        self.args = args
        self.device = device if device else args.device
        self.rank = rank

        self.model = model

        self.n_iter = 0
        self.num_epochs = args.num_epochs
        self.loss_func = loss_func
        self.optimizer = optimizer

        self.train_data = train_data
        self.dev_data = dev_data

        self.writer = SummaryWriter(log_dir=args.log_dir)
        self.model_out_path = args.model_out_path
        if not os.path.exists(self.model_out_path):
            os.makedirs(self.model_out_path)

    def train(self):
        self.model.to(self.device)
        self.model.train()

        # self.eval()
        tmp = float('inf')
        for epoch in range(self.num_epochs):
            tmp = self.train_one_epoch(tmp)
            # self.save()
            # self.eval()

    def train_one_epoch(self, tmp):
        TQDM = tqdm(self.train_data,ncols=80)
        best_loss = tmp
        dev_loss = tmp
        for sent, labels,_ in TQDM:
            loss = self.pipeline(sent, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # get train loss
            if self.rank == 0 and self.n_iter % self.args.train_loss_per_iter == 0:
                self.writer.add_scalar('Loss/train', loss, self.n_iter)
                # print(f'\ntrain loss: {float(loss):.3}')

            # get eval loss and save model
            if self.n_iter % self.args.eval_per_iter == 0:
                dev_loss = self.eval()
            if best_loss > dev_loss:
                best_loss = dev_loss
                self.save(name = 'best')
            if self.n_iter % self.args.save_per_iter == 0:
                self.save(name = 'iter')

            self.n_iter += 1
        return best_loss

    def eval(self):
        loss = 0
        data_size = len(self.dev_data)
        self.model.eval()
        TQDM = tqdm(self.dev_data,ncols=80)
        with torch.no_grad():
            for sent, labels,_ in TQDM:
                _loss = self.pipeline(sent, labels)
                loss += float(_loss.to('cpu'))
        loss /= data_size

        if self.rank == 0:
            print(f'iter: {self.n_iter}\tdev loss: {loss:.3f}')
            self.writer.add_scalar('Loss/dev', loss, self.n_iter)
        self.model.train()
        return loss
    
    def metric(self):
        
        data_l = self.train_data if self.train_data else self.dev_data
        TQDM = tqdm(data_l,ncols=80)
        with torch.no_grad():
            TP, TN, FN, FP = 0, 0, 0, 0
            for sent, labels,sent_o in TQDM:
                outputs = self.model(sent)
                outputs = torch.sigmoid(outputs)
                outputs = outputs.to('cpu')
                TP += sum([1 for s,l in zip(outputs, labels) if s > self.args.threshold and l == 1 ])
                FP += sum([1 for s,l in zip(outputs, labels) if s > self.args.threshold and l == 0 ])
                FN += sum([1 for s,l in zip(outputs, labels) if s <= self.args.threshold and l == 1 ])
                TN += sum([1 for s,l in zip(outputs, labels) if s <= self.args.threshold and l == 0 ])
            p = TP / (TP + FP)
            r = TP / (TP + FN)
            print(f"TP:{TP},TN:{TN},FN:{FN},FP:{FP}")
            F1 = 2 * r * p / (r + p)
            acc = (TP + TN) / (TP + TN + FP + FN)
        return f"F1:{F1},acc:{acc}"

    def pipeline(self, sent, labels):
        for k in sent:
            sent[k] = sent[k].to(self.device)
        labels = labels.to(self.device)
        outputs = self.model(sent)
        _loss = [self.loss_func(o, l) for o, l in zip(outputs, labels)]
        loss = sum(_loss)

        return loss

    def save(self,name):
        if self.rank == 0:
            if name == 'iter':
                torch.save(self.model.module.state_dict(),
                        f'{self.model_out_path}/checkpoint{self.n_iter}.pt')
            else:
                torch.save(self.model.module.state_dict(),
                        f'{self.model_out_path}/checkpoint_bestf1.pt')
