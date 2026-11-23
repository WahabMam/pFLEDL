import time
from flcore.clients.clientpolyak import clientPolyak
from flcore.servers.serverbase import Server


class FedPolyak(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientPolyak)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
    
    def train(self):
        for i in range(self.global_rounds+1):
            s_t = self.start_round_timer()
            self.selected_clients = self.select_clients()

            for client in self.clients:
                # client.set_parameters(self.global_model, force=i == self.global_rounds)
                client.set_parameters(
                    model=self.global_model, 
                    force=
                        i>self.global_rounds-self.args.freq*self.args.local_agg_times and 
                        i%self.args.freq==0
                )

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()
                client.polyak.add_weight(model=client.model)

            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            self.Budget.append(self.finish_round_timer(s_t))
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))
        
        # print(f'{self.clients[0].iter_choosen = }')

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientPolyak)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()