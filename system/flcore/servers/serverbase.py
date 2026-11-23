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
import csv
import os
import numpy as np
import h5py
import copy
import time
import random
from utils.data_utils import read_client_data
from utils.dlg import DLG
from utils.data_utils import get_global_test_data
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchmetrics.classification import MulticlassCalibrationError #MAK


class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.save_folder_name = args.save_folder_name
        self.top_cnt = 100
        self.auto_break = args.auto_break
        self.early_stop_patience = args.early_stop_patience
        self.early_stop_min_rounds = args.early_stop_min_rounds
        self.early_stop_min_delta = args.early_stop_min_delta
        if self.early_stop_patience < 1 or self.early_stop_min_rounds < 1 or self.early_stop_min_delta < 0:
            raise ValueError("Early stopping requires positive patience/minimum rounds and nonnegative min_delta")

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []
        self.rs_test_ece = [] #MAK
        self.rs_y_prob = [] #MAK
        self.rs_y_true = [] #MAK
        
        self.cen_test_acc = [] #MAK
        self.cen_test_auc = [] #MAK 
        self.cen_test_loss = [] #MAK
        self.cen_test_ece = [] #MAK
        self.cen_y_prob = [] #MAK
        self.cen_y_true = [] #MAK

        self.global_test_loader = self.load_global_test_data()

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        self.dlg_eval = args.dlg_eval
        self.dlg_gap = args.dlg_gap
        self.batch_num_per_client = args.batch_num_per_client

        self.num_new_clients = args.num_new_clients
        self.new_clients = []
        self.eval_new_clients = False
        self.fine_tuning_epoch_new = args.fine_tuning_epoch_new

    def set_clients(self, clientObj):
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=train_slow, 
                            send_slow=send_slow)
            self.clients.append(client)

    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)

    def select_clients(self):
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))

        return selected_clients

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            
            client.set_parameters(self.global_model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    # def aggregate_parameters(self):
    #     assert (len(self.uploaded_models) > 0)

    #     self.global_model = copy.deepcopy(self.uploaded_models[0])
    #     for param in self.global_model.parameters():
    #         param.data.zero_()
            
    #     for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
    #         self.add_parameters(w, client_model)
            
    #     self.evaluate_global_model_cen()


    def aggregate_parameters(self):  # Added by Wahab for FedPer, Babu, FedRep checking after getting, this 'ValueError: If `preds` have one dimension more than `target`, `preds.shape[1]` should be equal to number of classes'
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()
            
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

        # Centralized global evaluation is only valid when the global model
        # directly outputs class logits of shape [batch_size, num_classes].
        self.evaluate_global_model_cen()

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def save_global_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        torch.save(self.global_model, model_path)

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)

    def model_exists(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        return os.path.exists(model_path)
        
    # def save_results(self):

        algo = self.dataset + "_" + self.algorithm
        result_path = "results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_acc)):
            algo = algo + "_" + self.goal + "_" + str(self.times)
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
                hf.create_dataset('rs_test_ece', data=self.rs_test_ece) #MAK

                # hf.create_dataset('cen_test_acc', data=self.cen_test_acc) #MAK
                # hf.create_dataset('cen_test_auc', data=self.cen_test_auc) #MAK
                # hf.create_dataset('cen_test_loss', data=self.cen_test_loss) #MAK
                # hf.create_dataset('cen_test_ece', data=self.cen_test_ece) #MAK

                # added by WM
                hf.create_dataset('cen_test_acc', data=self.cen_test_acc)
                hf.create_dataset('cen_test_auc', data=self.cen_test_auc)
                hf.create_dataset('cen_test_loss', data=self.cen_test_loss)
                hf.create_dataset('cen_test_ece', data=self.cen_test_ece)

                if len(self.cen_y_prob) > 0:
                    hf.create_dataset('cen_y_prob', data=np.stack(self.cen_y_prob))
                    hf.create_dataset('cen_y_true', data=np.stack(self.cen_y_true))
                else:
                    hf.create_dataset(
                        'cen_y_prob',
                        data=np.empty((0, 0, self.num_classes), dtype=np.float32)
                    )
                    hf.create_dataset(
                        'cen_y_true',
                        data=np.empty((0, 0), dtype=np.int64)
                    )


                hf.create_dataset('y_prob', data=np.stack(self.rs_y_prob))  # shape [R, N, C]
                hf.create_dataset('y_true', data=np.stack(self.rs_y_true))  # shape [R, N]
                hf.create_dataset('cen_y_prob', data=np.stack(self.cen_y_prob))  # shape [R, N, C]
                hf.create_dataset('cen_y_true', data=np.stack(self.cen_y_true))
                
    def start_round_timer(self):
        """Measure complete rounds, including queued GPU work and evaluation."""
        if torch.device(self.device).type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def finish_round_timer(self, start):
        if torch.device(self.device).type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter() - start

    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        round_times = np.asarray(getattr(self, "Budget", []), dtype=float)
        cumulative_times = np.cumsum(round_times)
        timing_path = os.path.join(
            result_path, f"{algo}_{self.goal}_{self.times}_timing.csv"
        )
        with open(timing_path, 'w', newline='') as timing_file:
            writer = csv.writer(timing_file)
            writer.writerow(['round', 'round_time_sec', 'cumulative_time_sec'])
            writer.writerows(zip(range(len(round_times)), round_times, cumulative_times))

        if len(self.rs_test_acc):
            algo = algo + "_" + self.goal + "_" + str(self.times)
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('round_time_sec', data=round_times)
                hf.create_dataset('cumulative_time_sec', data=cumulative_times)
                hf.attrs['eval_gap'] = self.eval_gap
                hf.attrs['round_timing_scope'] = 'training, communication, aggregation and evaluation'
                # Personalized/client-side metrics
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
                hf.create_dataset('rs_test_ece', data=self.rs_test_ece)

                # Centralized/global metrics
                # These may be empty for FedPer/FedRep/FedProto-style methods
                # because their global model may not directly output class logits.
                hf.create_dataset('cen_test_acc', data=self.cen_test_acc)
                hf.create_dataset('cen_test_auc', data=self.cen_test_auc)
                hf.create_dataset('cen_test_loss', data=self.cen_test_loss)
                hf.create_dataset('cen_test_ece', data=self.cen_test_ece)

                # Personalized predictions
                if len(self.rs_y_prob) > 0:
                    hf.create_dataset('y_prob', data=np.stack(self.rs_y_prob))
                    hf.create_dataset('y_true', data=np.stack(self.rs_y_true))
                else:
                    hf.create_dataset(
                        'y_prob',
                        data=np.empty((0, 0, self.num_classes), dtype=np.float32)
                    )
                    hf.create_dataset(
                        'y_true',
                        data=np.empty((0, 0), dtype=np.int64)
                    )

                # Centralized/global predictions
                # For FedPer, this will be empty because centralized global eval is skipped.
                if len(self.cen_y_prob) > 0:
                    hf.create_dataset('cen_y_prob', data=np.stack(self.cen_y_prob))
                    hf.create_dataset('cen_y_true', data=np.stack(self.cen_y_true))
                else:
                    hf.create_dataset(
                        'cen_y_prob',
                        data=np.empty((0, 0, self.num_classes), dtype=np.float32)
                    )
                    hf.create_dataset(
                        'cen_y_true',
                        data=np.empty((0, 0), dtype=np.int64)
                    )           

    def save_item(self, item, item_name):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def load_item(self, item_name):
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def test_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()
        
        num_samples = []
        tot_correct = []
        tot_auc = []
        tot_ece = [] #MAK
        all_y_prob = []
        all_y_true = []
        for c in self.clients:
            ct, ns, auc, ece, y_prob, y_true = c.test_metrics() #MAK
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            tot_ece.append(ece) #MAK
            all_y_prob.append(y_prob)
            all_y_true.append(y_true)
            num_samples.append(ns)
            
        tot_y_prob = np.concatenate(all_y_prob, axis=0)
        tot_y_true = np.concatenate(all_y_true, axis=0)

        ids = [c.id for c in self.clients]
        return ids, num_samples, tot_correct, tot_auc, tot_ece, tot_y_prob, tot_y_true

    def train_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]
        
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    # evaluate selected clients
    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        test_auc = sum(stats[3])*1.0 / sum(stats[1])
        test_ece = sum(stats[4]) * 1.0 / len(stats[4]) #MAK
        self.rs_test_ece.append(test_ece) #MAK
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]
        eces = [a / n for a, n in zip(stats[4], stats[1])]
        self.rs_y_prob.append(stats[5])
        self.rs_y_true.append(stats[6])
        
        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
        
        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        print("Averaged Test ECE: {:.4f}".format(test_ece))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))
        print("Std Test ECE: {:.4f}".format(np.std(eces))) #MAK
    
    def load_global_test_data(self, batch_size=None):
        """
        Loads global test data from all clients and returns a DataLoader.
        """
        if batch_size is None:
            batch_size = self.batch_size
        test_data = get_global_test_data(self.dataset)
        return DataLoader(test_data, batch_size=batch_size, drop_last=False, shuffle=True) 

    def print_(self, test_acc, test_auc, train_loss):
        print("Average Test Accurancy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        """Stop on a test-accuracy plateau; patience is measured in global rounds."""
        completed = len(self.Budget)
        if completed < self.early_stop_min_rounds or not acc_lss:
            return False
        for values in acc_lss:
            if not values:
                return False
            best = -float('inf')
            last_improvement_round = 0
            for index, value in enumerate(values):
                if not np.isfinite(value):
                    return False
                if value > best + self.early_stop_min_delta:
                    best = value
                    last_improvement_round = index * self.eval_gap
            if completed - 1 - last_improvement_round < self.early_stop_patience:
                return False
        print(f"Early stopping after {completed} completed rounds: "
              f"no test-accuracy improvement greater than {self.early_stop_min_delta} "
              f"for {self.early_stop_patience} rounds.")
        return True

    def call_dlg(self, R):
        # items = []
        cnt = 0
        psnr_val = 0
        for cid, client_model in zip(self.uploaded_ids, self.uploaded_models):
            client_model.eval()
            origin_grad = []
            for gp, pp in zip(self.global_model.parameters(), client_model.parameters()):
                origin_grad.append(gp.data - pp.data)

            target_inputs = []
            trainloader = self.clients[cid].load_train_data()
            with torch.no_grad():
                for i, (x, y) in enumerate(trainloader):
                    if i >= self.batch_num_per_client:
                        break

                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    output = client_model(x)
                    target_inputs.append((x, output))

            d = DLG(client_model, origin_grad, target_inputs)
            if d is not None:
                psnr_val += d
                cnt += 1
            
            # items.append((client_model, origin_grad, target_inputs))
                
        if cnt > 0:
            print('PSNR value is {:.2f} dB'.format(psnr_val / cnt))
        else:
            print('PSNR error')

        # self.save_item(items, f'DLG_{R}')

    def set_new_clients(self, clientObj):
        for i in range(self.num_clients, self.num_clients + self.num_new_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=False, 
                            send_slow=False)
            self.new_clients.append(client)

    # fine-tuning on new clients
    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            client.set_parameters(self.global_model)
            opt = torch.optim.SGD(client.model.parameters(), lr=self.learning_rate)
            CEloss = torch.nn.CrossEntropyLoss()
            trainloader = client.load_train_data()
            client.model.train()
            for e in range(self.fine_tuning_epoch_new):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(client.device)
                    else:
                        x = x.to(client.device)
                    y = y.to(client.device)
                    output = client.model(x)
                    loss = CEloss(output, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

    # evaluating on new clients
    def test_metrics_new_clients(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.new_clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.new_clients]

        return ids, num_samples, tot_correct, tot_auc

    def evaluate_global_model_cen(self):
        testloaderfull = self.global_test_loader
        self.global_model.eval()

        total_correct = 0
        total_samples = 0
        y_prob = []
        y_true = []
        total_loss = 0.0
        criterion = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for x, y in testloaderfull:
                if isinstance(x, (list, tuple)):
                    x = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                output = self.global_model(x)

                # Important fix:
                # Skip centralized evaluation if the global model does not output class logits.
                if output.dim() != 2 or output.shape[1] != self.num_classes:
                    print(
                    f"[Warning] Skipping centralized global evaluation for {self.algorithm}: "
                    f"global model output shape is {tuple(output.shape)}, "
                    f"but num_classes={self.num_classes}. "
                    f"This is expected for FedPer/FedRep-style personalized-head methods."
                )
                return


                probs = F.softmax(output, dim=1)
                y_prob.append(probs.cpu())
                y_true.append(y.cpu())

                loss = criterion(output, y)
                total_loss += loss.item() * y.size(0)

                total_correct += (torch.argmax(output, dim=1) == y).sum().item()
                total_samples += y.size(0)

        y_prob = torch.cat(y_prob, dim=0)
        y_true = torch.cat(y_true, dim=0)

        # auc = metrics.roc_auc_score(...) # your auc code here
        auc = 0

        # ECE calculation
        ece_metric = MulticlassCalibrationError(num_classes=self.num_classes, n_bins=15, norm='l1')
        ece = ece_metric(y_prob, y_true).item()

        accuracy = total_correct / total_samples if total_samples > 0 else 0
        avg_loss = total_loss / total_samples if total_samples > 0 else 0

        self.cen_test_acc.append(accuracy)
        self.cen_test_auc.append(auc)
        self.cen_test_ece.append(ece)
        self.cen_test_loss.append(avg_loss)
        self.cen_y_prob.append(y_prob.numpy())
        self.cen_y_true.append(y_true.numpy())