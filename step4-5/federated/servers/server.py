import copy
import numpy as np
import torch
from collections import OrderedDict

from torch import optim


class Server:

    def __init__(self, model, writer, local_rank, lr, momentum, optimizer, source_dataset):
        self.model = copy.deepcopy(model)
        self.model_params_dict = copy.deepcopy(self.model.state_dict())
        self.writer = writer
        self.selected_clients = []
        self.updates = []
        self.local_rank = local_rank
        self.opt_string = optimizer
        self.lr = lr
        self.momentum = momentum
        self.optimizer = self.__get_optimizer()
        self.total_grad = 0
        self.source_dataset = source_dataset
        self.swa_model = None

    def __get_optimizer(self):

        if self.opt_string is None:
            self.writer.write("Running without server optimizer")
            return None

        if self.opt_string == 'SGD':
            return optim.SGD(params=self.model.parameters(), lr=self.lr, momentum=self.momentum)

        if self.opt_string == 'FedAvgm':
            return optim.SGD(params=self.model.parameters(), lr=1, momentum=0.9)

        if self.opt_string == 'Adam':
            return optim.Adam(params=self.model.parameters(), lr=self.lr, betas=(0.9, 0.99), eps=10 ** (-1))

        if self.opt_string == 'AdaGrad':
            return optim.Adagrad(params=self.model.parameters(), lr=self.lr, eps=10 ** (-2))

        raise NotImplementedError

    def select_clients(self, my_round, possible_clients, num_clients):
        num_clients = min(num_clients, len(possible_clients))
        np.random.seed(my_round)
        self.selected_clients = np.random.choice(possible_clients, num_clients, replace=False)

    def get_clients_info(self, clients):
        if clients is None:
            clients = self.selected_clients
        num_samples = {c.id: c.num_samples for c in clients}
        return num_samples

    @staticmethod
    def num_parameters(params):
        return sum(p.numel() for p in params if p.requires_grad)

    def train_source(self, *args, **kwargs):
        raise NotImplementedError

    def _compute_client_delta(self, cmodel):
        delta = OrderedDict.fromkeys(cmodel.keys())
        for k, x, y in zip(self.model_params_dict.keys(), self.model_params_dict.values(), cmodel.values()):
            delta[k] = y - x if "running" not in k and "num_batches_tracked" not in k else y
        return delta

    def train_clients(self, partial_metric=None, r=None, metrics=None, target_test_client=None, test_interval=None,
                      ret_score='Mean IoU'):

        if self.optimizer is not None:
            self.optimizer.zero_grad()

        clients = self.selected_clients
        losses = {}

        for i, c in enumerate(clients):

            self.writer.write(f"CLIENT {i + 1}/{len(clients)}: {c}")

            c.model.load_state_dict(self.model_params_dict)
            out = c.train(partial_metric, r=r)

            if self.local_rank == 0:
                num_samples, update, dict_losses_list = out
                losses[c.id] = {'loss': dict_losses_list, 'num_samples': num_samples}
            else:
                num_samples, update = out

            if self.optimizer is not None:
                update = self._compute_client_delta(update)

            self.updates.append((num_samples, update))

        if self.local_rank == 0:
            return losses
        return None

    def _aggregation(self):
        total_weight = 0.
        base = OrderedDict()
        for (client_samples, client_model) in self.updates:
            total_weight += client_samples
            for key, value in client_model.items():
                if key in base:
                    base[key] += client_samples * value.type(torch.FloatTensor)
                else:
                    base[key] = client_samples * value.type(torch.FloatTensor)
        averaged_sol_n = copy.deepcopy(self.model_params_dict)
        for key, value in base.items():
            if total_weight != 0:
                averaged_sol_n[key] = value.to(self.local_rank) / total_weight
        return averaged_sol_n

    def _server_opt(self, pseudo_gradient):
        for n, p in self.model.named_parameters():
            p.grad = -1.0 * pseudo_gradient[n]
        self.optimizer.step()
        bn_layers = \
            OrderedDict({k: v for k, v in pseudo_gradient.items() if "running" in k or "num_batches_tracked" in k})
        self.model.load_state_dict(bn_layers, strict=False)

    def _get_model_total_grad(self):
        total_norm = 0
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_grad = total_norm ** 0.5
        self.writer.write(f"total grad norm: {round(total_grad, 2)}")
        return total_grad

    def update_model(self):

        averaged_sol_n = self._aggregation()

        if self.optimizer is not None:
            self._server_opt(averaged_sol_n)
            self.total_grad = self._get_model_total_grad()
        else:
            self.model.load_state_dict(averaged_sol_n)
        self.model_params_dict = copy.deepcopy(self.model.state_dict())

        self.updates = []


    def setup_swa_model(self, swa_ckpt=None):
        self.swa_model = copy.deepcopy(self.model)
        if swa_ckpt is not None:
            self.swa_model.load_state_dict(swa_ckpt)

    def update_swa_model(self, alpha):
        for param1, param2 in zip(self.swa_model.parameters(), self.model.parameters()):
            param1.data *= (1.0 - alpha)
            param1.data += param2.data * alpha
