import numpy as np
import time
from flcore.clients.clientbase import Client
from utils.Polyak import Polyak

class clientPolyak(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.iter = 0

        # polyak
        self.global_rounds = args.global_rounds
        self.mix = args.mix
        self.freq = args.freq
        self.polyak = Polyak(mix=self.mix, model=self.model, freq=self.freq)


    def train(self):
        trainloader = self.load_train_data()
        self.model.train()
        
        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for _ in range(max_local_epochs):
            for x, y in trainloader:
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
    
    def set_parameters(self, model, force=False):
        if force:
            model = self.polyak.models_mixture(old=self.model, new=model)
            print(f'Client {self.id} - Ites chosen: {self.polyak.iter_choosen}')
        super().set_parameters(model)