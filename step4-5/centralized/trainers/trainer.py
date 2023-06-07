import torch

from general_trainer import GeneralTrainer
from utils import get_optimizer_and_scheduler
from metrics import StreamSegMetrics

class Trainer(GeneralTrainer):

    def __init__(self, args, writer, device, rank, world_size):
        super().__init__(args, writer, device, rank, world_size)

    def server_setup(self):
        return None

    @staticmethod
    def set_metrics(writer, num_classes):
        writer.write("Setting up metrics...")
        metrics = {
            'test': StreamSegMetrics(num_classes, 'test'),
            'train': StreamSegMetrics(num_classes, 'train'),
        }
        writer.write("Done.")
        return metrics

    def max_iter(self):
        return self.args.num_epochs * self.target_train_clients[0].len_loader

    def get_optimizer_and_scheduler(self):
        return get_optimizer_and_scheduler(self.args, self.model.parameters(), self.max_iter())

    def load_from_checkpoint(self):
        self.model.load_state_dict(self.checkpoint["model_state"])
        self.writer.write(f"[!] Model restored from step {self.checkpoint_step}.")
        self.optimizer.load_state_dict(self.checkpoint["optimizer_state"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(self.checkpoint["scheduler_state"])
        self.writer.write(f"[!] Optimizer and scheduler restored.")

    def save_model(self, step, optimizer=None, scheduler=None):
        state = {
            "step": step,
            "model_state": self.model.state_dict(),
            "optimizer_state": optimizer.state_dict()
        }
        if scheduler is not None:
            state["scheduler_state"] = scheduler.state_dict()
        torch.save(state, self.ckpt_path)
        self.writer.wandb.save(self.ckpt_path)

    def handle_ckpt_step(self):
        return None, None, self.checkpoint_step, None

    def perform_fed_oracle_training(self, partial_train_metric, eval_train_metric, test_metric, max_scores=None):

        if max_scores is None:
            max_scores = [0] * len(self.target_test_clients)

        for r in range(self.ckpt_round, self.args.num_rounds):

            self.writer.write(f'ROUND {r + 1}/{self.args.num_rounds}: '
                              f'Training {self.args.clients_per_round} Clients...')
            self.server.select_clients(r, self.target_train_clients, num_clients=self.args.clients_per_round)
            losses = self.server.train_clients(partial_metric=partial_train_metric)
            self.plot_train_metric(r, partial_train_metric, losses)
            partial_train_metric.reset()

            self.server.update_model()
            self.model.load_state_dict(self.server.model_params_dict)
            self.save_model(r + 1, optimizer=self.server.optimizer)

            if (r + 1) % self.args.eval_interval == 0 and \
                    self.all_target_client.loader.dataset.ds_type not in ('unsupervised',):
                self.test([self.all_target_client], eval_train_metric, r, 'ROUND', self.get_fake_max_scores(False, 1),
                          cl_type='target')
            if (r + 1) % self.args.test_interval == 0 or (r + 1) == self.args.num_rounds:
                max_scores, _ = self.test(self.target_test_clients, test_metric, r, 'ROUND', max_scores,
                                          cl_type='target')

        return max_scores

    def train(self):
        return self.perform_fed_oracle_training(
            partial_train_metric=self.metrics['partial_train'],
            eval_train_metric=self.metrics['eval_train'],
            test_metric=self.metrics['test']
        )
