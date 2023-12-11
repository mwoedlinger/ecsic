__all__ = []
__author__ = "Matthias Wödlinger"

import sa.models as models
from sa import *
from train import experiment
import numpy as np
import wandb
from datetime import datetime
from pathlib import Path
import argparse
import json
import inspect


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Order of arguments: gpu_idx exp_name')
    parser.add_argument('argv', nargs='*', help='exp_name gpu_idx')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--config', type=str)
    parser.add_argument('--project', type=str)
    parser.add_argument('--entity', type=str)
    parser.add_argument('--tags', type=str, help='comma separated list of tags')
    parser.add_argument('--lmda', type=str, help='comma separated list of lambda values')
    parser.add_argument('--lr', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr_drop', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--model', type=str, help=f'Options: {", ".join([l[0] for l in inspect.getmembers(models, inspect.isclass) if l[1].__module__ == "sa.models"])}')
    parser.add_argument('--train', help=f'name of training dataset. Options: {", ".join(list_datasets())}', type=str)
    parser.add_argument('--eval', help=f'name of training dataset. Options: {", ".join(list_datasets())}', type=str)
    parser.add_argument('--test', help=f'name of training dataset. Options: {", ".join(list_datasets())}', type=str)
    args = parser.parse_args()

    if args.debug:
        args.lmda = "0.01, 0.1"

    config = process_cli_arguments(args, print_config=False)

    dt = str(datetime.now())
    run_name = str(config.exp_name) + '-' + dt[2:10].replace('-', '') + '-' + dt[11:19].replace(':', '')
    lmbda_values = [float(l) for l in (config.lmda).replace(' ', '').split(',')]

    results = {'lambda': []}
    for l in lmbda_values:
        config.lmda = l
        config.run_id = wandb.util.generate_id()

        exp_path = Path(f'./experiments/RD_curves') / run_name / str(l)
        exp_path.mkdir(exist_ok=False, parents=True)
        config.exp_path = str(exp_path)
         
        tags = ['RD', str(l), (args.train or str(config.train.name)), config.model.name, Path(args.config).stem]
        if args.tags:
            tags.append(args.tags.replace(' ', '').split(','))
        config.tags = tags

        print('CONFIG:')
        pprint(config)
        print('\n')

        save_config(config, exp_path)

        # Initialize wandb
        wandb.init(group=config.exp_name, project=config.project, entity=config.entity, tags=config.tags, config=config, id=config.run_id, resume="allow")
        wandb.run.log_code('.')
        wandb.run.name = config.run_id
        
        s = f'#### Start run with λ = {l} ####'
        print(len(s)*'#' + '\n' + s + '\n' + len(s)*'#')
        test_results = experiment(config)
        print(f'finished training {l}')

        results['lambda'].append(l)
        for k, v in test_results.items():
            if k not in results:
                results[k] = [np.mean(v)]
            else:
                results[k].append(np.mean(v))

        wandb.finish(quiet=True)
        
    with open(str(Path(f'./experiments/RD_curves') / run_name / 'results.json'), 'w') as f:
        json.dump(results, f)
        