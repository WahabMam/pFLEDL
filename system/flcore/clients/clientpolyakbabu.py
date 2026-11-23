import numpy as np
import time
from flcore.clients.clientbase import Client


class clientPolyakBABU(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.fine_tuning_epochs = args.fine_tuning_epochs

        for param in self.model.head.parameters():
            param.requires_grad = False
        
        # polyak
        args.mix = 0.9
        args.tail_length = 50
        self.polyak = TailedPolyak(mix=args.mix, model=self.model.base, tail_length=args.tail_length)


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

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        self.polyak.add_weight(model=self.model.base)

    def set_parameters(self, base):
        if self.polyak.num_weights == 0:
            for new_param, old_param in zip(base.parameters(), self.model.base.parameters()):
                old_param.data = new_param.data.clone()
        else:
            self.model.base = self.polyak.models_mixture(old=self.model.base, new=base)
        

    def fine_tune(self, which_module=['base', 'head']):
        trainloader = self.load_train_data()
        
        start_time = time.time()
        
        self.model.train()

        if 'head' in which_module:
            for param in self.model.head.parameters():
                param.requires_grad = True

        if 'base' not in which_module:
            for param in self.model.head.parameters():
                param.requires_grad = False
            

        for epoch in range(self.fine_tuning_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.train_time_cost['total_cost'] += time.time() - start_time

import copy

class PolyakBase:
    def __init__(self):
        self.num_weights = 0

    def polyak_ruppert_averaging(self):
        pass 

    def models_mixture(self, old, new):
        for new_param, old_param, avg in zip(new.parameters(), old.parameters(), self.polyak_ruppert_averaging().parameters()):
            old_param.data = new_param.data.clone() - self.mix * (new_param.data.clone() - avg.data.clone())  
        return old

    def divide_constant(self, model, divisor):
        result = copy.deepcopy(model)
        for r, param in zip(result.parameters(), model.parameters()):
            r.data = param.data.clone() / divisor
        return result
    
    def add_parameters(self, model1, model2):
        result = copy.deepcopy(model1)
        for r, param1, param2 in zip(result.parameters(), model1.parameters(), model2.parameters()):
            r.data = param1.data.clone() + param2.data.clone()
        return result
    
    def subtract_parameters(self, model1, model2):
        result = copy.deepcopy(model1)
        for r, param1, param2 in zip(result.parameters(), model1.parameters(), model2.parameters()):
            r.data = param1.data.clone() - param2.data.clone()
        return result

    def zero_parameters(self, model):
        zero = copy.deepcopy(model)
        for param in zero.parameters():
            param.data.zero_()
        return zero

class Polyak(PolyakBase):
    def __init__(self, mix, model):
        super().__init__()
        self.mix = mix
        self.weight_sum = self.zero_parameters(model)
    
    def add_weight(self, model):
        self.num_weights += 1
        self.weight_sum = self.add_parameters(
            self.weight_sum, 
            model
        )
    
    def polyak_ruppert_averaging(self):
        return self.divide_constant(self.weight_sum, self.num_weights)

class TailedPolyak(PolyakBase):
    def __init__(self, mix, model, tail_length):
        super().__init__()
        self.mix = mix
        self.tail_length = tail_length
        self.prefix_sum = []  
        self.prefix_sum.append(self.zero_parameters(model))
    
    def add_weight(self, model):
        self.num_weights += 1
        self.prefix_sum.append(
            self.add_parameters(
                self.prefix_sum[-1], 
                model
            )
        )
    
    def polyak_ruppert_averaging(self):
        if len(self.prefix_sum) <= self.tail_length:
            sum_parameters = self.prefix_sum[-1]
        else:
            sum_parameters = self.subtract_parameters(
                self.prefix_sum[-1], 
                self.prefix_sum[-(self.tail_length + 1)]
            )  
            self.prefix_sum.pop(0)
        
        averaged_model = self.divide_constant(
            sum_parameters, 
            min(
                len(self.prefix_sum)-1, 
                self.tail_length
            )
        )
        return averaged_model