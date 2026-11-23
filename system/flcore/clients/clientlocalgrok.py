import time

import numpy as np
import torch
import torch.nn.functional as F

from flcore.clients.clientavg import clientAVG
from utils.data_utils import read_client_data


class clientLocalGrok(clientAVG):
    """clientAVG, but reconfigured to match Omnigrok's (arXiv 2210.01117) exact recipe:
    MSE loss on one-hot targets (not CrossEntropy), AdamW with explicit weight_decay
    (not plain SGD), and a large multiplicative init-scale alpha applied to the model's
    standard (Kaiming-uniform) initialization -- alpha is the central lever the paper
    identifies for inducing grokking on real data; without it grokking does not appear.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        if getattr(args, "grok_full_batch", False):
            # True full-batch GD on this client's own shard, regardless of shard
            # size. Avoids the mini-batch-SGD-noise confound that a single global
            # -lbs value introduces under Dirichlet/pathological partitions, where
            # per-client shard sizes are unequal.
            self.batch_size = self.train_samples

        init_scale = getattr(args, "init_scale", 1.0)
        if init_scale != 1.0:
            with torch.no_grad():
                for p in self.model.parameters():
                    p.mul_(init_scale)

        self.loss = torch.nn.MSELoss()
        weight_decay = getattr(args, "weight_decay", 0.0)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=weight_decay
        )

    def load_train_data(self, batch_size=None):
        # read_client_data() re-reads the client's .npz off disk on every call;
        # clientAVG's train()/train_metrics() call it every single round. The
        # underlying data never changes, so cache it once per client -- pure I/O
        # optimization, DataLoader still reshuffles fresh each call, no change to
        # batching/optimization semantics.
        if batch_size is None:
            batch_size = self.batch_size
        if not hasattr(self, "_train_data_cache"):
            self._train_data_cache = read_client_data(self.dataset, self.id, is_train=True)
        return torch.utils.data.DataLoader(
            self._train_data_cache, batch_size, drop_last=True, shuffle=True)

    def load_test_data(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        if not hasattr(self, "_test_data_cache"):
            self._test_data_cache = read_client_data(self.dataset, self.id, is_train=False)
        return torch.utils.data.DataLoader(
            self._test_data_cache, batch_size, drop_last=False, shuffle=True)

    def _onehot(self, y):
        return F.one_hot(y, num_classes=self.num_classes).float()

    def weight_norm(self):
        total = 0.0
        for p in self.model.parameters():
            total += p.detach().pow(2).sum().item()
        return total ** 0.5

    def train(self):
        trainloader = self.load_train_data()
        self.model.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, self._onehot(y))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def train_metrics(self):
        trainloader = self.load_train_data()
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, self._onehot(y))
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num

    def train_metrics_with_acc(self):
        trainloader = self.load_train_data()
        self.model.eval()

        train_acc = 0
        train_num = 0
        train_loss = 0.0
        with torch.no_grad():
            for x, y in trainloader:
                x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                train_acc += torch.sum(torch.argmax(output, dim=1) == y).item()
                train_loss += self.loss(output, self._onehot(y)).item() * y.shape[0]
                train_num += y.shape[0]

        return train_acc, train_loss, train_num

    def test_metrics_with_loss(self):
        testloaderfull = self.load_test_data()
        self.model.eval()

        test_acc = 0
        test_num = 0
        test_loss = 0.0
        with torch.no_grad():
            for x, y in testloaderfull:
                x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_loss += self.loss(output, self._onehot(y)).item() * y.shape[0]
                test_num += y.shape[0]

        return test_acc, test_loss, test_num
