# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import torch
import numpy as np
import time
import copy
from flcore.optimizers.fedoptimizer import PerturbedGradientDescent
from flcore.clients.clientbase import Client
import torch.nn.functional as F
from sklearn.preprocessing import label_binarize
from sklearn import metrics

from torchmetrics.classification import MulticlassCalibrationError #MAK


class clientDitto(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.mu = args.mu
        self.plocal_epochs = args.plocal_epochs

        self.model_per = copy.deepcopy(self.model)
        self.optimizer_per = PerturbedGradientDescent(
            self.model_per.parameters(), lr=self.learning_rate, mu=self.mu)
        self.learning_rate_scheduler_per = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_per, 
            gamma=args.learning_rate_decay_gamma
        )

    def train(self):
        trainloader = self.load_train_data()
        
        start_time = time.time()

        # self.model.to(self.device)
        self.model.train()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
            self.learning_rate_scheduler_per.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        
    def ptrain(self):
        trainloader = self.load_train_data()

        start_time = time.time()

        # self.model.to(self.device)
        self.model_per.train()

        max_local_epochs = self.plocal_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model_per(x)
                loss = self.loss(output, y)
                self.optimizer_per.zero_grad()
                loss.backward()
                self.optimizer_per.step(self.model.parameters(), self.device)

        # self.model.cpu()

        self.train_time_cost['total_cost'] += time.time() - start_time

    def test_metrics_personalized(self):
        testloaderfull = self.load_test_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model_per.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        
        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model_per(x)

                probs = F.softmax(output, dim=1)                # shape: [batch_size, num_classes]
                y_prob.append(probs.cpu())                      # collect as tensor
                y_true.append(y.cpu())                          # collect as tensor

                test_acc += (torch.argmax(output, dim=1) == y).sum().item()
                test_num += y.size(0)

        y_prob = torch.cat(y_prob, dim=0)                      # shape [N, num_classes]
        y_true = torch.cat(y_true, dim=0)                      # shape [N]


        # auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        auc = 0
         # ECE calculation
        ece_metric = MulticlassCalibrationError(num_classes=self.num_classes, n_bins=15, norm='l1')
        ece = ece_metric(y_prob, y_true).item()

        
        return test_acc, test_num, auc, ece, y_prob, y_true

    def train_metrics_personalized(self):
        trainloader = self.load_train_data()
        self.model_per.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model_per(x)
                loss = self.loss(output, y)

                gm = torch.cat([p.data.view(-1) for p in self.model.parameters()], dim=0)
                pm = torch.cat([p.data.view(-1) for p in self.model_per.parameters()], dim=0)
                loss += 0.5 * self.mu * torch.norm(gm-pm, p=2)
                
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num