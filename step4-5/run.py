import os
import sys
import time
import warnings

from utils import dynamic_import
from utils import dist_utils, setup_env


def run_experiment(args):
    writer, device, rank, world_size = setup_env(args)

    trainer_class = dynamic_import(args.framework, args.fw_task, 'trainer')
    trainer = trainer_class(args, writer, device, rank, world_size)

    writer.write("The experiment begins...")
    max_score = trainer.train(*trainer.train_args, **trainer.train_kwargs)
    writer.write("Training completed.")

    writer.write(f"Final mIoU: {round(max_score[0] * 100, 3)}%")

    if args.random_seed is not None:
        dist_utils.cleanup()
