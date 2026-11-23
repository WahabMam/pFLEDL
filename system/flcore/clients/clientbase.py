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

import copy
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data

# from utils.ece_utils import ECEEvaluator # Wahab
import torch.nn.functional as F # Wahab

from torchmetrics.classification import MulticlassCalibrationError #MAK







class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        self.model = copy.deepcopy(args.model)
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer
        self.save_folder_name = args.save_folder_name

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs

        # self.ece_evaluator = ECEEvaluator(num_classes=self.num_classes, device= self.device) # Wahab 


        self.has_BatchNorm = False
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay




    #  universal ECE+Accuracy evaluation method 

    def evaluate_ece_and_accuracy(self):
        self.model.eval()
        all_logits = []
        all_labels = []
        testloader = self.load_test_data()

        with torch.no_grad():
            for x, y in testloader:
                if isinstance(x, (list, tuple)):
                    x = x[0]
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x)
                all_logits.append(logits)
                all_labels.append(y)

        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)

        # Accuracy calculation
        probs = F.softmax(all_logits, dim=1)
        predicted = torch.argmax(probs, dim=1)
        correct = (predicted == all_labels).sum().item()
        accuracy = correct / len(all_labels)

        # ECE calculation
        ece_value = self.ece_evaluator.compute_ece(all_logits, all_labels)

        # Print or store metrics
        if hasattr(self, 'verbose') and self.verbose:
            print(f"Client {self.id} - Test Accuracy: {accuracy:.4f}, ECE: {ece_value:.4f}")

        return accuracy, ece_value





    # check BatchNorm

    


    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, is_train=False)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)
        
    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    # def test_metrics(self):
    #     testloaderfull = self.load_test_data()
    #     # self.model = self.load_model('model')
    #     # self.model.to(self.device)
    #     self.model.eval()

    #     test_acc = 0
    #     test_num = 0
    #     y_prob = []
    #     y_true = []
    #     y_true_non_binary = [] #MAK
        
    #     with torch.no_grad():
    #         for x, y in testloaderfull:
    #             if type(x) == type([]):
    #                 x[0] = x[0].to(self.device)
    #             else:
    #                 x = x.to(self.device)
    #             y = y.to(self.device)
    #             output = self.model(x)

    #             test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
    #             test_num += y.shape[0]

    #             y_prob.append(output.detach().cpu().numpy())
    #             nc = self.num_classes
    #             if self.num_classes == 2:
    #                 nc += 1
    #             lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
    #             if self.num_classes == 2:
    #                 lb = lb[:, :2]
    #             y_true.append(lb)
    #             y_true_non_binary.append(y.detach().cpu().numpy()) #MAK

    #     # self.model.cpu()
    #     # self.save_model(self.model, 'model')

    #     y_prob = np.concatenate(y_prob, axis=0)
    #     y_true = np.concatenate(y_true, axis=0)
    #     # auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
    #     auc = 0
        
    #      # ECE calculation
    #     y_prob = torch.tensor(np.concatenate(y_prob, axis=0))    # Convert to tensor
    #     y_true = torch.tensor(np.concatenate(y_true_non_binary, axis=0))    # Convert to tensor
    #     ece_metric = MulticlassCalibrationError(num_classes=self.num_classes, n_bins=15, norm='l1') #MAK
    #     ece = ece_metric(y_prob, y_true).item() #MAK
        
    #     # return test_acc, test_num, auc #MAK
    #     return test_acc, test_num, auc, ece #MAK
    def test_metrics(self):
        testloaderfull = self.load_test_data()
        self.model.eval()

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
                output = self.model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]
                oo = output.detach().cpu().numpy()
                y_prob.append(oo)
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        # self.model.cpu()
        # self.save_model(self.model, 'model')
        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        # auc = metrics.roc_auc_score(...) # your auc code here
        auc = 0


        # ECE calculation
        # Convert numpy arrays to torch tensors
        y_prob_tensor = torch.tensor(y_prob, dtype=torch.float32)
        y_true_tensor = torch.tensor(np.argmax(y_true, axis=1), dtype=torch.long)
        ece_metric = MulticlassCalibrationError(num_classes=self.num_classes, n_bins=15, norm='l1')
        ece = ece_metric(y_prob_tensor, y_true_tensor).item()

        return test_acc, test_num, auc, ece, y_prob, y_true

    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num

    # def get_next_train_batch(self):
    #     try:
    #         # Samples a new batch for persionalizing
    #         (x, y) = next(self.iter_trainloader)
    #     except StopIteration:
    #         # restart the generator if the previous generator is exhausted.
    #         self.iter_trainloader = iter(self.trainloader)
    #         (x, y) = next(self.iter_trainloader)

    #     if type(x) == type([]):
    #         x = x[0]
    #     x = x.to(self.device)
    #     y = y.to(self.device)

    #     return x, y


    def save_item(self, item, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    def load_item(self, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        return torch.load(os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    # @staticmethod
    # def model_exists():
    #     return os.path.exists(os.path.join("models", "server" + ".pt"))
