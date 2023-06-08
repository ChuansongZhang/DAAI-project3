import time
import importlib


def run_experiment(args):
  
    if args.framework == 'federated':
        main_module = 'fed_setting.main'
        main = getattr(importlib.import_module(main_module), 'main')
        main(args)
    elif args.framework == 'centralized':
        main_module = 'centr_setting.main'
        main = getattr(importlib.import_module(main_module), 'main')
        main(args)
    else:
        raise NotImplementedError


