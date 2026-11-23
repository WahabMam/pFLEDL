import time

import torch

from flcore.clients.clientlocalgrok import clientLocalGrok
from flcore.servers.servergrokbase import GrokServerBase


class FedAvgGrok(GrokServerBase):
    """FedAvg (real weight-averaging aggregation each round), using clientLocalGrok
    (Omnigrok recipe) and the same grok CSV logger as LocalGrok -- directly comparable.
    Tests whether FedAvg's per-round weight-averaging (which pulls weight norm down,
    the same direction grokking's slow phase moves in) accelerates, delays, or
    destroys grokking, versus the single-client LocalGrok positive control.
    """

    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_slow_clients()
        self.set_clients(clientLocalGrok)

        # send_models() broadcasts self.global_model to every client each round,
        # including round 0 -- if left at Server.__init__'s standard (unscaled) init,
        # that broadcast would silently overwrite the large init_scale each client's
        # constructor just applied, before a single training step ever ran.
        init_scale = getattr(args, "init_scale", 1.0)
        if init_scale != 1.0:
            with torch.no_grad():
                for p in self.global_model.parameters():
                    p.mul_(init_scale)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []
        self._init_grok_logging(args)

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = self.start_round_timer()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                self._log_round(i)

            for client in self.selected_clients:
                client.train()

            self.receive_models()
            self.aggregate_parameters()

            self.Budget.append(self.finish_round_timer(s_t))
            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        if self.global_rounds % self.eval_gap != 0:
            self._log_round(self.global_rounds)
        self.save_results()
        self.save_global_model()
