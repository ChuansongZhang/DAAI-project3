from metrics import StreamSegMetrics
from .trainer import Trainer


class SourceOnlyTrainer(Trainer):

    def __init__(self, args, writer, device, rank, world_size):
        super().__init__(args, writer, device, rank, world_size)

    @staticmethod
    def set_metrics(writer, num_classes):
        writer.write("Setting up metrics...")
        metrics = {
            'train': StreamSegMetrics(num_classes, 'train'),
            'test_source': StreamSegMetrics(num_classes, 'test_source'),
            'test_target': StreamSegMetrics(num_classes, 'test_target'),
        }
        writer.write("Done.")
        return metrics

    def handle_ckpt_step(self):
        return None, self.checkpoint_step, None, None

    def max_iter(self):
        return self.args.num_source_epochs * self.source_train_clients[0].len_loader

    def perform_centr_sourceonly_training(self, e_type, e, e_ref, train_client, train_metric):
        self.writer.write(f"{e_type}: {e + 1}/{e_ref}")
        self.model.train()
        _ = train_client.run_epoch(e, self.optimizer, train_metric, self.scheduler, e_name=e_type)
        # train_metric.synch(self.device)
        self.writer.plot_metric(e, train_metric, str(train_client), self.ret_score)
        train_metric.reset()
        self.save_model(e + 1, self.optimizer, self.scheduler)

    def train(self, train_metric=None, max_scores=None):

        if train_metric is None:
            train_metric = self.metrics['train']

        max_scores = [0] * len(self.target_test_clients)

        for e in range(self.ckpt_source_epoch, self.args.num_source_epochs):
            self.perform_centr_sourceonly_training('SOURCE EPOCH', e, self.args.num_source_epochs,
                                               self.source_train_clients[0], train_metric)
            if (e + 1) % self.args.test_interval == 0 or (e + 1) == self.args.num_source_epochs:
                max_scores, improvement = self.test(self.target_test_clients, self.metrics['test_target'], e,
                                                    'SOURCE EPOCH', max_scores, cl_type='target')
                self.test(self.source_test_clients, self.metrics['test_source'], e, 'SOURCE EPOCH',
                          self.get_fake_max_scores(improvement, len(self.source_test_clients)), cl_type='source')

        return max_scores
