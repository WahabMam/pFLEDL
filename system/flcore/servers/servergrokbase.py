import csv
import os

from flcore.servers.serverbase import Server


class GrokServerBase(Server):
    """Shared CSV logging for the grokking probe servers (LocalGrok, FedAvgGrok, ...).
    Tracks weight_decay and init_scale explicitly per row so every run's hyperparameters
    are traceable directly from the CSV, not just from the directory name.
    """

    def _init_grok_logging(self, args):
        self.weight_decay = getattr(args, "weight_decay", 0.0)
        self.init_scale = getattr(args, "init_scale", 1.0)
        self.shrink_mode = getattr(args, "grok_shrink_mode", "none")
        self.shrink_rho = getattr(args, "grok_shrink_rho", 1.0)

        csv_dir = os.path.join(self.save_folder_name, "grok_csv",
                                f"{self.dataset}_{self.algorithm}_{self.goal}")
        os.makedirs(csv_dir, exist_ok=True)
        self.csv_path = os.path.join(csv_dir, f"grok_{self.times}.csv")
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                ["round", "train_loss", "train_acc", "test_loss", "test_acc",
                 "weight_norm", "weight_decay", "init_scale", "shrink_mode",
                 "shrink_rho", "shrink_active", "trigger_round"]
            )

    def _log_round(self, round_num):
        train_num_tot = 0
        train_loss_tot = 0.0
        train_acc_tot = 0
        test_num_tot = 0
        test_loss_tot = 0.0
        test_acc_tot = 0
        norm_sq_tot = 0.0

        for c in self.clients:
            train_acc, cl, ns = c.train_metrics_with_acc()
            train_acc_tot += train_acc
            train_loss_tot += cl
            train_num_tot += ns

            acc, tloss, tn = c.test_metrics_with_loss()
            test_acc_tot += acc
            test_loss_tot += tloss
            test_num_tot += tn

            norm_sq_tot += c.weight_norm() ** 2

        train_loss = train_loss_tot / train_num_tot
        train_acc = train_acc_tot / train_num_tot
        test_loss = test_loss_tot / test_num_tot
        test_acc = test_acc_tot / test_num_tot
        weight_norm = (norm_sq_tot / len(self.clients)) ** 0.5

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [round_num, train_loss, train_acc, test_loss, test_acc, weight_norm,
                 self.weight_decay, self.init_scale, self.shrink_mode, self.shrink_rho,
                 getattr(self, "shrink_active", False),
                 getattr(self, "trigger_round", -1)]
            )

        self.rs_test_acc.append(test_acc)
        self.rs_train_loss.append(train_loss)

        print(f"round {round_num}: train_loss={train_loss:.6g} train_acc={train_acc:.4f} "
              f"test_loss={test_loss:.6g} test_acc={test_acc:.4f} "
              f"weight_norm={weight_norm:.4f}")

        return {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "weight_norm": weight_norm,
        }
