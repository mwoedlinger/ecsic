import torch
import wandb
import json
import numpy as np
import torch.nn.functional as F
from os import system, getcwd
from sys import exit
from time import time
from datetime import datetime
from pathlib import Path
from shutil import copytree, copy
import os
import math
from .transforms import *

__all__ = ['Dict', 'save_model', 'load_model', 'save_config', 'load_config', 'pprint', 'get_lr', 'aij',
           'load_config_from_json', 'process_cli_arguments', 'get_positional_encoding', 'get_positional_fourier_encoding']

def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']

def aij(w, L, k):
    """
    Expects w \in R^(L_w x d_w)
    """
    assert len(w) >= 2*k + 1, f'w is too short for given context length {k}'
    i = torch.linspace(0, L-1, L).unsqueeze(1).long()
    j = torch.linspace(0, L-1, L).unsqueeze(0).long()
    idx_clipped = torch.clip((i-j), min=-k, max=k)+k
    
    return w[idx_clipped]

class Dict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        return self.__setitem__(key, value)


def ToDict(dictionary: dict):
    for key, value in dictionary.items():
        if isinstance(value, dict):
            dictionary[key] = ToDict(value)

    return Dict(dictionary)

def tictoc():
    """
    Returns time in seconds since the last time the function was called.
    For the initial call 0 is returned.
    """
    if not hasattr(tictoc, 'tic'):
        tictoc.tic = time()

    toc = time()
    dt = toc - tictoc.tic
    tictoc.tic = toc

    return dt

def quantise(x, mean, training):
    """ 
    Performs quantisation of the latent `x` for the entropy calculation and decoder.


    Returns:
        torch.Tensor: x_entropy, if training==True: noise quantised tensor, else STE
        torch.Tensor: x_decoder, STE quantised tensor
    """
    if training:
        # in training it's split such that entropy uses noise, decoder uses STE
        x_entropy = x + torch.zeros_like(x, device=x.device).uniform_(-0.5, 0.5)
        x_ste = x - mean
        x_ste = x_ste + (torch.round(x_ste) - x_ste).detach()
        x_ste = x_ste + mean
        return x_entropy, x_ste 
    else:
        # in validation both use STE
        x_ste = x - mean
        x_ste = x_ste + (torch.round(x_ste) - x_ste).detach()
        x_ste = x_ste + mean
        x_entropy, x_decoder = x_ste, x_ste
        return x_entropy, x_decoder

def get_positional_encoding(x):
    _, h, w = x.shape

    dx = torch.linspace(0, w-1, w).repeat((h, 1)) - (w//2)
    dy = torch.linspace(0, h-1, h).unsqueeze(1).repeat((1, w)) - (h//2)
    return torch.stack((dx, dy), dim=0)

def get_positional_fourier_encoding(h, w, d=32):
    div_term = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
    
    dx = torch.arange(w).unsqueeze(1)/w
    pe_x = torch.zeros(w, d)
    pe_x[:, 0::2] = torch.sin(dx * div_term)
    pe_x[:, 1::2] = torch.cos(dx * div_term)
    
    dy = torch.arange(h).unsqueeze(1)/h
    pe_y = torch.zeros(h, d)
    pe_y[:, 0::2] = torch.sin(dy * div_term)
    pe_y[:, 1::2] = torch.cos(dy * div_term)
    
    pe_x = pe_x.unsqueeze(0).permute((2,0,1)).repeat((1, h, 1))
    pe_y = pe_y.unsqueeze(0).permute((2,1,0)).repeat((1, 1, w))
    
    return torch.cat([pe_x, pe_y], dim=0)

def save_model(path, model, optimizer, scheduler, epoch, suffix=''):
    state_dict_model = model.state_dict()
    state_dict_training = {
        'epoch': epoch,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }

    print(f'save model in {path}')
    model_name = f'model{suffix}.pt'
    training_state_name = f'training_state{suffix}.pt'

    torch.save(state_dict_model, str(Path(path) / model_name))
    torch.save(state_dict_training, str(Path(path) / training_state_name))


def load_model(path, model, optimizer=None, scheduler=None, device=torch.device('cpu')):
    checkpoint_model = torch.load(str(Path(path) / 'model.pt'), map_location=device)
    checkpoint_training = torch.load(str(Path(path) / 'training_state.pt'), map_location=device)

    model.load_state_dict(checkpoint_model)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_training['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint_training['scheduler'])

    return checkpoint_training['epoch']


def save_config(config, path):
    config_simple = ToDict(config.copy())

    config_simple.train.transform = None
    config_simple.eval.transform = None
    config_simple.test.transform = None
    config_simple.device = config.device.index
    config_simple.exp_path = config.exp_path
    config_simple.resume = config.resume

    with open(str(Path(path) / 'config.json'), 'w') as file:
        json.dump(config_simple, file)


def load_config(path: str):
    with open(str(Path(path) / 'config.json'), 'r') as file:
        config = ToDict(json.load(file))

    config.device = torch.device(f'cuda:{config.device}')
    config.exp_path = config.exp_path

    return config


def load_config_from_json(path):
    with open(path, 'r') as file:
        config = json.load(file)

    return ToDict(config)


def update_dict(config, config_new):
    for k, v in config_new.items():
        config[k] = v


def pprint(c, indent=0):
    if indent == 0:
        print('-'*45)

    if c.keys():
        offset = max([len(k) for k in c.keys()] + [15]) + 2

    for k, v in c.items():
        spaces = offset - len(str(k)) - 1
        if isinstance(v, Dict) or isinstance(v, dict):
            print(' '*indent + str(k) + ':')
            pprint(v, indent + offset)
        elif isinstance(v, list):
            print(' '*indent + str(k) + ':' + ' '*spaces + '[')
            for l in v:
                print(' '*(indent + offset + 1) + str(l))
            print(' '*(indent + offset) + ']')
        else:
            print(' '*indent + str(k) + ':' + ' '*spaces + str(v))
    if indent == 0:
        print('-'*45)


def copy_files(exp_path, root='.', src='sa', data='data'):
    dst = Path(exp_path) / 'src'
    dst.mkdir()

    copytree(src, dst / src)
    copytree(data, dst / data)

    print(data)
    print(dst / data)

    remaining_files = [f for f in Path(root).iterdir() if f.suffix == '.py']

    for file in remaining_files:
        file_dst = dst / file.name
        copy(file, file_dst)


def process_cli_arguments(args, default_config=Path('./configs/default.json'), print_config=True):
    if not default_config.is_file(): # for resume
        default_config = Path('../../../') / default_config

    # Priorities: args > loaded_config > default
    # Meaning if for any value the command line arguments are used if available, if not the preloaded config is used, if neither is available use default parameters.
    assert len(args.argv) > 0, 'You need to specifiy the gpu index'

    log_dir = args.log_dir
    if not Path(log_dir).exists:
        print(f'log_dir {log_dir} does not exist, creating it.')
        Path(log_dir).mkdir(parents=True)

    # This is only for generate_rd_curve.py
    if 'resume' not in args:
        args.__dict__.update(
            {'resume': None, 'resume_new': None, 'testing': None})

    if args.resume:
        resume = str(Path(__file__).parent.parent.parent)
        config = load_config(resume)
        exp_path = config.exp_path

    elif args.resume_new:
        # if resume_new update dict and create new exp_dir and wandb run
        resume = str(Path(__file__).parent.parent.parent)
        config = load_config(resume)
        config.run_id = wandb.util.generate_id()

        dt = str(datetime.now())
        run_name = str(config.run_id) + '-' + \
            dt[2:10].replace('-', '') + '-' + dt[11:19].replace(':', '')
        exp_path = str(Path(log_dir) / Path('experiments') / run_name)
        Path(exp_path).mkdir(exist_ok=False, parents=True)
        copy_files(exp_path, root=(Path(resume) / 'src'))
    
    else:
        # else create new exp_dir and wandb run
        resume = False
        config = load_config_from_json(default_config)
        config.run_id = wandb.util.generate_id()

        dt = str(datetime.now())
        run_name = str(config.run_id) + '-' + \
            dt[2:10].replace('-', '') + '-' + dt[11:19].replace(':', '')
        exp_path = str(Path(log_dir) / Path('experiments') / run_name)
        Path(exp_path).mkdir(exist_ok=False, parents=True)
        copy_files(exp_path)

    args.resume = resume
    if args.config:
        update_dict(config, load_config_from_json(args.config))

    if args.debug:
        config.epochs = 1
        config.lr_drop = 20
        config.eval_steps = 15
        config.train.debug = True
        config.eval.debug = True
        config.test.debug = True

    if len(args.argv) > 1:
        config.exp_name = args.argv[1]

    if args.testing:
        args.epochs = 0

    config.exp_path = exp_path
    config.device = torch.device(f'cuda:{args.argv[0]}')
    for arg, value in args.__dict__.items():
        if value is None:
            continue

        if arg in ['testing', 'resume_new', 'tags', 'argv']:
            pass
        elif arg == 'train':
            config.train.name = value
        elif arg == 'eval':
            config.eval.name = value
        elif arg == 'test':
            config.test.name = value
        elif arg == 'model':
            config.model.name = value
        else:
            config[arg] = value

    dataset_specific_transform = {
        'cityscapes': TWrapper(CropCityscapesArtefacts()),
        'instereo2k': TWrapper(MinimalCrop(32))
    }
    config.train.transform = [
        dataset_specific_transform.get(config.train.name, TWrapper(lambda x: x)), 
        RandomCrop((256, 1024))
        ]
    config.eval.transform = config.test.transform = [
        dataset_specific_transform.get(config.eval.name, TWrapper(lambda x: x))
        ]
    config.test.transform = [
        dataset_specific_transform.get(config.test.name, TWrapper(lambda x: x))
        ]

    if args.tags is not None:
        config.tags = args.tags.replace(' ', '').split(',') + config.tags

    if config.exp_name is None:
        config.exp_name = config.model.name

    # fix random seeds for reproducibility
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(config.seed)

    if print_config:
        print('CONFIG:')
        pprint(config)
        print('\n')

    save_config(config, exp_path)

    # Set wandb logging and cache dir
    wandb_cache_dir = Path(log_dir) / Path('wandb') / Path('.cache')
    wandb_dir = Path(log_dir) / Path('wandb') / config.project / config.exp_name / config.run_id
    if not Path(wandb_dir).is_dir():
        Path(wandb_dir).mkdir(parents=True)
    if not Path(wandb_cache_dir).is_dir():
        Path(wandb_cache_dir).mkdir(parents=True)

    assert Path(wandb_cache_dir).is_dir(), f'wandb_cache_dir {wandb_cache_dir} does not exist/is no dir.'
    assert Path(wandb_dir).is_dir(), f'wandb_dir {wandb_dir} does not exist/is no dir.'

    os.environ['WANDB_CACHE_DIR'] = str(wandb_cache_dir)
    os.environ['WANDB_DIR'] = str(wandb_dir)

    return config
