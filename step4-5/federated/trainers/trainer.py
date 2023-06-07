import copy
import torch

from utils import dynamic_import, weight_train_loss
from general_trainer import GeneralTrainer
from metrics import StreamSegMetrics

class Trainer(GeneralTrainer):

    def __init__(self, args, writer, device, rank, world_size):
        super().__init__(args, writer, device, rank, world_size)
        self.all_target_client = self.gen_all_target_client()

    def gen_all_target_client(self):
        client_class = dynamic_import(self.args.framework, self.args.fw_task, 'client')
        cl_args = {**self.clients_shared_args, **self.clients_args['all_train'][0]}
        return client_class(**cl_args, batch_size=self.args.test_batch_size, test_user=True)

    def server_setup(self):
        server_class = dynamic_import(self.args.framework, self.args.fw_task, 'server')
        server = server_class(self.model, self.writer, self.args.local_rank, self.args.server_lr,
                              self.args.server_momentum, self.args.server_opt, self.args.source_dataset)
        return server

    @staticmethod
    def set_metrics(writer, num_classes):
        writer.write("Setting up metrics...")
        metrics = {
            'test': StreamSegMetrics(num_classes, 'test'),
            'partial_train': StreamSegMetrics(num_classes, 'partial_train'),
            'eval_train': StreamSegMetrics(num_classes, 'eval_train')
        }
        writer.write("Done.")
        return metrics

    def get_optimizer_and_scheduler(self):
        return None, None

    def load_from_checkpoint(self):
        self.model.load_state_dict(self.checkpoint["model_state"])
        self.server.model_params_dict = copy.deepcopy(self.model.state_dict())
        self.writer.write(f"[!] Model restored from step {self.checkpoint_step}.")
        if "server_optimizer_state" in self.checkpoint.keys():
            self.server.optimizer.load_state_dict(self.checkpoint["server_optimizer_state"])
            self.writer.write(f"[!] Server optimizer restored.")

    def save_model(self, step, optimizer=None, scheduler=None):
        state = {
            "step": step,
            "model_state": self.server.model_params_dict
        }
        if self.server.optimizer is not None:
            state["server_optimizer_state"] = self.server.optimizer.state_dict()
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

    def train(self, *args, **kwargs):
        return self.perform_fed_oracle_training(
            partial_train_metric=self.metrics['partial_train'],
            eval_train_metric=self.metrics['eval_train'],
            test_metric=self.metrics['test']
        )

    def plot_train_metric(self, r, metric, losses, plot_metric=True):
        if self.args.local_rank == 0:
            round_losses = weight_train_loss(losses)
            self.writer.plot_step_loss(metric.name, r, round_losses)
            if plot_metric:
                self.writer.plot_metric(r, metric, '', self.ret_score)
