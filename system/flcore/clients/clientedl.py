# # PFLlib: Personalized Federated Learning Algorithm Library
# # Copyright (C) 2021  Jianqing Zhang

# # This program is free software; you can redistribute it and/or modify
# # it under the terms of the GNU General Public License as published by
# # the Free Software Foundation; either version 2 of the License, or
# # (at your option) any later version.

# # This program is distributed in the hope that it will be useful,
# # but WITHOUT ANY WARRANTY; without even the implied warranty of
# # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# # GNU General Public License for more details.

# # You should have received a copy of the GNU General Public License along
# # with this program; if not, write to the Free Software Foundation, Inc.,
# # 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

# import copy
# import torch
# import numpy as np
# import time
# from flcore.clients.clientbase import Client
# import torch
# import torch.nn as nn
# import random
# from torch.utils.data import DataLoader
# from utils.data_utils import read_client_data

# class clientEDL(Client):
#     def __init__(self, args, id, train_samples, test_samples, **kwargs):
#         super().__init__(args, id, train_samples, test_samples, **kwargs)
#         self.rand_percent = 80
#         self.le = 1
#         self.warm_up = 30
#         self.lam = 0.01
#         self.lam_decay = 0.9
#         self.lam_interval = 10
#         self.fintune_loss = EDL_Loss(self.num_classes).to(self.device)
#         self.iter = 0
#     def train(self):
#         trainloader = self.load_train_data()
#         # self.model.to(self.device)
#         self.model.train()
        
#         start_time = time.time()

#         max_local_epochs = self.local_epochs
#         if self.train_slow:
#             max_local_epochs = np.random.randint(1, max_local_epochs // 2)

#         for epoch in range(max_local_epochs):
#             for i, (x, y) in enumerate(trainloader):
#                 if type(x) == type([]):
#                     x[0] = x[0].to(self.device)
#                 else:
#                     x = x.to(self.device)
#                 y = y.to(self.device)
#                 if self.train_slow:
#                     time.sleep(0.1 * np.abs(np.random.rand()))
#                 output = self.model(x)
#                 loss = self.loss(output, y)
#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 self.optimizer.step()

#         # self.model.cpu()

#         if self.learning_rate_decay:
#             self.learning_rate_scheduler.step()

#         self.train_time_cost['num_rounds'] += 1
#         self.train_time_cost['total_cost'] += time.time() - start_time

#     def local_initialization(self, received_global_model):
#         self.iter += 1
#         # if self.iter % self.lam_interval == 0:
#             # self.lam *= self.lam_decay
        
#         # print(self.lam)

#         if self.iter >= self.warm_up:
#             train_data = read_client_data(self.dataset, self.id, is_train=True)
#             rand_ratio = self.rand_percent / 100
#             rand_num = int(rand_ratio*len(train_data))
#             rand_idx = random.randint(0, len(train_data)-rand_num)
#             rand_loader = DataLoader(train_data[rand_idx:rand_idx+rand_num], self.batch_size, drop_last=False)

#             self.set_parameters(received_global_model)
#             self.model.train()
#             self.model.to(self.device)
#             for _ in range(self.le):
#                 for i, (x, y) in enumerate(rand_loader):
#                     x = x.to(self.device)
#                     y = y.to(self.device)
#                     output = self.model(x)
#                     loss = self.fintune_loss(output, y, self.lam)
#                     self.optimizer.zero_grad()
#                     loss.backward()
#                     self.optimizer.step()


# class EDL_Loss(nn.Module):
#     """
#     evidence deep learning loss
#     https://github.com/chenchenzong/DPC/blob/main/AAAI2024_DPC_code/loss.py
#     """
#     def __init__(self, num_classes):
#         super(EDL_Loss, self).__init__()
#         self.num_classes = num_classes

#     def forward(self, logits, labels, lam):
#         alpha = torch.exp(logits)+10./self.num_classes
#         total_alpha = torch.sum(alpha, dim=1, keepdim=True)  # total_alpha.shape: [B, 1]

#         # one_hot_y = torch.eye(logits.shape[1])
#         one_hot_y = torch.eye(logits.shape[1]).cuda()
#         one_hot_y.requires_grad = False
#         loss_nll = torch.sum(one_hot_y * (total_alpha.log() - alpha.log()), dim=1) # / logits.shape[0]

#         # uniform_bata = torch.ones((1, logits.shape[1]))
#         uniform_bata = torch.ones((1, logits.shape[1])).cuda()
#         uniform_bata.requires_grad = False
#         total_uniform_beta = torch.sum(uniform_bata, dim=1)
#         new_alpha = one_hot_y + (1.0 - one_hot_y) * (self.num_classes / 10.) * alpha
#         new_total_alpha = torch.sum(new_alpha, dim=1)  # new_total_alpha.shape: [B]
#         loss_kl = torch.lgamma(new_total_alpha) - torch.lgamma(total_uniform_beta) - torch.sum(torch.lgamma(new_alpha), dim=1) \
#             + torch.sum((new_alpha - 1) * (torch.digamma(new_alpha) - torch.digamma(new_total_alpha.unsqueeze(1))), dim=1)
#         loss_kl = 0.5*loss_kl / self.num_classes

#         return loss_nll.mean()+ loss_kl.mean()
############################################################


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

# import copy
# import torch
# import numpy as np
# import time
# from flcore.clients.clientbase import Client
# import torch
# import torch.nn as nn
# import random
# from torch.utils.data import DataLoader
# from utils.data_utils import read_client_data



# class clientEDL(Client):
#     def __init__(self, args, id, train_samples, test_samples, **kwargs):
#         super().__init__(args, id, train_samples, test_samples, **kwargs)
#         self.rand_percent = 100
#         self.le = 5
#         self.warm_up = 30
#         self.lam = 0.01
#         self.lam_decay = 0.9
#         self.lam_interval = 10
#         self.fintune_loss = EDL_Loss(self.num_classes).to(self.device)
#         self.iter = 0
#         self.verbose = True
#     def train(self):
#         trainloader = self.load_train_data()
#         # self.model.to(self.device)
#         self.model.train()
        
#         start_time = time.time()

#         max_local_epochs = self.local_epochs
#         if self.train_slow:
#             max_local_epochs = np.random.randint(1, max_local_epochs // 2)

#         for epoch in range(max_local_epochs):
#             for i, (x, y) in enumerate(trainloader):
#                 if type(x) == type([]):
#                     x[0] = x[0].to(self.device)
#                 else:
#                     x = x.to(self.device)
#                 y = y.to(self.device)
#                 if self.train_slow:
#                     time.sleep(0.1 * np.abs(np.random.rand()))
#                 output = self.model(x)
#                 loss = self.loss(output, y)
#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 self.optimizer.step()

#         # self.model.cpu()

#         if self.learning_rate_decay:
#             self.learning_rate_scheduler.step()

#         self.train_time_cost['num_rounds'] += 1
#         self.train_time_cost['total_cost'] += time.time() - start_time

#     def local_initialization(self, received_global_model):
#         self.iter += 1
#         # if self.iter % self.lam_interval == 0:
#             # self.lam *= self.lam_decay
        
#         # print(self.lam)

#         if self.iter >= self.warm_up:
#             train_data = read_client_data(self.dataset, self.id, is_train=True)
#             rand_ratio = self.rand_percent / 100
#             rand_num = int(rand_ratio*len(train_data))
#             rand_idx = random.randint(0, len(train_data)-rand_num)
#             rand_loader = DataLoader(train_data[rand_idx:rand_idx+rand_num], self.batch_size, drop_last=False)

#             self.set_parameters(received_global_model)
#             self.model.train()
#             self.model.to(self.device)
#             for _ in range(self.le):
#                 for i, (x, y) in enumerate(rand_loader):
#                     x = x.to(self.device)
#                     y = y.to(self.device)
#                     output = self.model(x)
#                     loss = self.fintune_loss(output, y, self.lam)
#                     self.optimizer.zero_grad()
#                     loss.backward()
#                     self.optimizer.step()


# class EDL_Loss(nn.Module):######÷

    # """
    # evidence deep learning loss
    # https://github.com/chenchenzong/DPC/blob/main/AAAI2024_DPC_code/loss.py
    # """
    # def __init__(self, num_classes):
    #     super(EDL_Loss, self).__init__()
    #     self.num_classes = num_classes

    # def forward(self, logits, labels, lam):
    #     alpha = torch.exp(logits)+10./self.num_classes
    #     total_alpha = torch.sum(alpha, dim=1, keepdim=True)  # total_alpha.shape: [B, 1]

    #     # one_hot_y = torch.eye(logits.shape[1])
    #     one_hot_y = torch.eye(logits.shape[1]).cuda()
    #     one_hot_y = one_hot_y[labels]
    #     one_hot_y.requires_grad = False
    #     loss_nll = torch.sum(one_hot_y * (total_alpha.log() - alpha.log()), dim=1) # / logits.shape[0]

    #     # uniform_bata = torch.ones((1, logits.shape[1]))
    #     uniform_bata = torch.ones((1, logits.shape[1])).cuda()
    #     uniform_bata.requires_grad = False
    #     total_uniform_beta = torch.sum(uniform_bata, dim=1)
    #     new_alpha = one_hot_y + (1.0 - one_hot_y) * (self.num_classes / 10.) * alpha
    #     new_total_alpha = torch.sum(new_alpha, dim=1)  # new_total_alpha.shape: [B]
    #     loss_kl = torch.lgamma(new_total_alpha) - torch.lgamma(total_uniform_beta) - torch.sum(torch.lgamma(new_alpha), dim=1) \
    #         + torch.sum((new_alpha - 1) * (torch.digamma(new_alpha) - torch.digamma(new_total_alpha.unsqueeze(1))), dim=1)
    #     loss_kl = 0.5*loss_kl / self.num_classes

    #     return loss_nll.mean()+ loss_kl.mean()

##################################################################################################################################################

import copy
import torch
import numpy as np
import time
from flcore.clients.clientbase import Client
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.utils.data import DataLoader
from utils.data_utils import read_client_data
# from utils.ece_utils import ECEEvaluator # Wahab 


class clientEDL(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.rand_percent = 50  # Use full client data for calibration
        self.le = 5     # Calibration epochs increased for deeper adaptation
        self.warm_up = 30
        self.lam = 0.01        # EDL regularization weight, 0.01 the main one used in experiments
        self.lam_decay = 0.9
        self.lam_interval = 10
        self.fintune_loss = EDL_Loss(self.num_classes).to(self.device)
        self.iter = 0
        self.beta = 0.02   # Fixed small calibration loss weight, 0.02 was the main one
        self.verbose = True      # Enable logging of calibration loss
        # self.ece_evalutor = ECEEvaluator(num_classes= self.num_classes, device = self.device) # Wahab 


    def evaluate(self): 
        return self.evaluate_ece_and_accuracy()  # Wahab
  

    def train(self):
        trainloader = self.load_train_data()
        self.model.train()

        start_time = time.time()
        
        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if isinstance(x, (list, tuple)):
                    x = x[0]
                x = x.to(self.device)
                y = y.to(self.device)

                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                output = self.model(x)
                loss = self.loss(output, y)  # Pure cross-entropy for main training
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def local_initialization(self, received_global_model, round_index=None):
        # Preserve the full-participation warm-up schedule when clients skip rounds.
        self.iter = self.iter + 1 if round_index is None else round_index + 1
        self.set_parameters(received_global_model)
        if self.iter < self.warm_up:
            return

        train_data = read_client_data(self.dataset, self.id, is_train=True)
        calib_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, drop_last=False)

        self.model.train()
        self.model.to(self.device)
        total_calib_loss = 0.0
        total_batches = 0

        for _ in range(self.le):
            for x, y in calib_loader:
                if isinstance(x, (list, tuple)):
                    x = x[0]
                x = x.to(self.device)
                y = y.to(self.device)

                output = self.model(x)
                loss = self.beta * self.fintune_loss(output, y, self.lam)
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"WARNING: Skipping batch with NaN or Inf loss during calibration on client {self.id}")
                    continue
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()

                total_calib_loss += loss.item()
                total_batches += 1

        if self.verbose and total_batches > 0:
            avg_loss = total_calib_loss / total_batches
            print(f"Round {self.iter}: Client {self.id} calibration loss: {avg_loss:.4f} (beta={self.beta})")


class EDL_Loss(nn.Module):
    """
    evidence deep learning loss
    https://github.com/chenchenzong/DPC/blob/main/AAAI2024_DPC_code/loss.py
    """
    def __init__(self, num_classes):
        super(EDL_Loss, self).__init__()
        self.num_classes = num_classes

    def forward(self, logits, labels, lam):
        alpha = torch.exp(logits)+10./self.num_classes
        total_alpha = torch.sum(alpha, dim=1, keepdim=True)  # total_alpha.shape: [B, 1]

        # one_hot_y = torch.eye(logits.shape[1])
        one_hot_y = torch.eye(logits.shape[1]).cuda()
        one_hot_y = one_hot_y[labels]
        one_hot_y.requires_grad = False
        loss_nll = torch.sum(one_hot_y * (total_alpha.log() - alpha.log()), dim=1) # / logits.shape[0]

        # uniform_bata = torch.ones((1, logits.shape[1]))
        uniform_bata = torch.ones((1, logits.shape[1])).cuda()
        uniform_bata.requires_grad = False
        total_uniform_beta = torch.sum(uniform_bata, dim=1)
        new_alpha = one_hot_y + (1.0 - one_hot_y) * (self.num_classes / 10.) * alpha
        new_total_alpha = torch.sum(new_alpha, dim=1)  # new_total_alpha.shape: [B]
        loss_kl = torch.lgamma(new_total_alpha) - torch.lgamma(total_uniform_beta) - torch.sum(torch.lgamma(new_alpha), dim=1) \
            + torch.sum((new_alpha - 1) * (torch.digamma(new_alpha) - torch.digamma(new_total_alpha.unsqueeze(1))), dim=1)
        loss_kl = 0.5*loss_kl / self.num_classes

        # return loss_nll.mean()+ loss_kl.mean()
        return loss_nll.mean() + lam * loss_kl.mean() # added Lam by WM
    
###########################################################################################
 
