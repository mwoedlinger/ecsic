__all__ = ['experiment', 'train', 'evaluation']
__author__ = "Matthias Wödlinger"

import sa.models as models
import sa.optimizers as optimizers
import sa.schedulers as schedulers
from sa import *

import torch
from torch.utils.data import DataLoader
from pytorch_msssim import ms_ssim as calc_ms_ssim
import numpy as np
import wandb
from tqdm import tqdm
import argparse
import inspect
from pathlib import Path

def experiment(CFG):
    """
    Training initialization and loop.
    """

    # Init dataloaders
    train_set = Dataset(data_type='train', **CFG.train, pos_encoding=CFG.model.kwargs.get('pos_encoding', False))
    train_loader = DataLoader(train_set, batch_size=CFG.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    train_logger = Logger(CFG.train.name + '#train')
                              
    eval_set = Dataset(data_type='eval', **CFG.eval, pos_encoding=CFG.model.kwargs.get('pos_encoding', False))
    eval_loader = DataLoader(eval_set, batch_size=1, shuffle=False,
                              num_workers=4, pin_memory=False)
    eval_logger = Logger(CFG.eval.name + '#eval')
                              
    test_set = Dataset(data_type='test', **CFG.test, pos_encoding=CFG.model.kwargs.get('pos_encoding', False))
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False,
                              num_workers=4, pin_memory=False)
    test_logger = Logger(CFG.test.name + '#test')

    # initialize model
    model = getattr(models, CFG.model.name)(**CFG.model.kwargs)
    model = model.to(CFG.device)
    # wandb.watch(model, log_freq=100, log='all')

    # init optimizer and scheduler
    optimizer = getattr(optimizers, CFG.optimizer.name)(model.parameters(), lr=CFG.lr, **CFG.optimizer.kwargs)
    scheduler = getattr(schedulers, CFG.scheduler.name)(optimizer, **CFG.scheduler.kwargs)

    # train model
    step = train(train_loader, eval_loader, model, train_logger, eval_logger, optimizer, scheduler, CFG)

    # test model after training
    print(f'\n## TESTING ON {test_logger.prefix} ##')
    with torch.no_grad():
        evaluation(test_loader, model, test_logger, CFG)
        test_results = test_logger.scal.copy()
        test_logger.log(step)
        
    return test_results

def train(train_loader, eval_loader, model, train_logger, eval_logger, optimizer, scheduler, CFG):
    """
    Training loop function.
    """
    if CFG.resume:
        print(f'Resume from {CFG.resume}')
        start_epoch = load_model(CFG.resume, model, optimizer, scheduler)
    else:
        start_epoch = 0

    # TRAINING LOOP
    step = start_epoch * len(train_loader.dataset)
    for epoch in range(start_epoch, CFG.epochs):
        print(f'\ntrain epoch {epoch}/{CFG.epochs}')

        for batch in tqdm(train_loader, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
            if len(batch) == 2:
                left, right = batch
                pos = None
            else:
                left, right, pos = batch

            B = left.size(0)
            _train_step(left, right, pos, model, optimizer, train_logger, step, CFG)
            if step % CFG.eval_steps < B:
                with torch.no_grad():
                    evaluation(eval_loader, model, eval_logger, CFG)
                    eval_logger.log(step)
                save_model(CFG.exp_path, model, optimizer, scheduler, epoch)
            if step % CFG.lr_drop < B and step > 0:
                save_model(CFG.exp_path, model, optimizer, scheduler, epoch, suffix=f'_{get_lr(optimizer)}')
                scheduler.step()

            step += B
        train_logger.log(step)

    return step

def _train_step(left, right, pos, model, optimizer, train_logger, step, CFG):
    """
    A single training step
    """
    model.train()

    left = left.to(CFG.device)
    right = right.to(CFG.device)
    pos = pos if pos is None else pos.to(CFG.device)

    optimizer.zero_grad()

    output = model(left, right, pos)
    pred, rate, latents = output.pred, output.rate, output.latents
    pred_left, pred_right = pred.left, pred.right

    # Compute MSE
    mse_left = calc_mse(left, pred_left)
    mse_right = calc_mse(right, pred_right)
    mse = (mse_left + mse_right)/2

    # Compute PSNR
    psnr_left = calc_psnr(mse_left, eps=CFG.eps)
    psnr_right = calc_psnr(mse_right, eps=CFG.eps)
    psnr = (psnr_left + psnr_right)/2

    # Computer BPP
    bpp_y_left = calc_bpp(rate.left.y, left)
    bpp_z_left = calc_bpp(rate.left.z, left)
    bpp_y_right = calc_bpp(rate.right.y, right)
    bpp_z_right = calc_bpp(rate.right.z, right)
    bpp = (bpp_y_left + bpp_z_left + bpp_y_right + bpp_z_right)/2
    bpp_y = bpp_y_left + bpp_y_right
    bpp_z = bpp_z_left + bpp_z_right

    # Computer RD-Loss
    loss = (bpp + CFG.lmda * mse) / (1 + CFG.lmda)

    # Backward - optimize
    loss.mean().backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.50)
    optimizer.step()

    # Log scalars
    train_logger.scalars(
        loss=loss, bpp=bpp, bpp_y=bpp_y, bpp_z=bpp_z, mse=mse, psnr=psnr, lr=get_lr(optimizer)
    )
    # train_logger.log(step)


def evaluation(eval_loader, model, eval_logger, CFG):
    """
    Evalution loop function.
    """
    for batch in tqdm(eval_loader, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
        if len(batch) == 2:
            left, right = batch
            pos = None
        else:
            left, right, pos = batch
        _evaluation_step(left, right, pos, model, eval_logger, CFG)
        
    log = [f'    ## {eval_logger.prefix} averages:']
    for name, vals in eval_logger.scal.items():
        log.append(f'    {name:10}: {np.mean(vals):.4}')
    log.append(f'tags = {CFG.tags}')
    log_str = '\n' + '\n'.join(log)


    with open(Path(CFG.exp_path) / 'log.txt', 'a') as f:
        f.write(log_str)
        print(log_str)

def _evaluation_step(left, right, pos, model, eval_logger, CFG):
    """
    A single evaluation step
    """
    model.eval()

    left = left.to(CFG.device)
    right = right.to(CFG.device)
    pos = pos if pos is None else pos.to(CFG.device)

    output = model(left, right, pos)
    pred, rate, latents = output.pred, output.rate, output.latents
    pred_left = torch.clamp(pred.left, min=0.0, max=1.0)
    pred_right = torch.clamp(pred.right, min=0.0, max=1.0)

    # Compute MSE
    mse_left = calc_mse(left, pred_left)
    mse_right = calc_mse(right, pred_right)
    mse = (mse_left + mse_right)/2

    # Compute PSNR
    psnr_left = calc_psnr(mse_left, eps=CFG.eps)
    psnr_right = calc_psnr(mse_right, eps=CFG.eps)
    psnr = (psnr_left + psnr_right)/2

    ms_ssim_left = calc_ms_ssim(left, pred_left)
    ms_ssim_right = calc_ms_ssim(right, pred_right)
    ms_ssim = (ms_ssim_left + ms_ssim_right)/2

    # Computer BPP
    bpp_y_left = calc_bpp(rate.left.y, left)
    bpp_z_left = calc_bpp(rate.left.z, left)
    bpp_y_right = calc_bpp(rate.right.y, right)
    bpp_z_right = calc_bpp(rate.right.z, right)
    bpp = (bpp_y_left + bpp_z_left + bpp_y_right + bpp_z_right)/2
    bpp_y = bpp_y_left + bpp_y_right
    bpp_z = bpp_z_left + bpp_z_right

    # Computer RD-Loss
    loss = (bpp + CFG.lmda * mse) / (1 + CFG.lmda)

    if CFG.log_images:
        images_gt = torch.cat([left, right], dim=-1)
        images_pred = torch.cat([pred.left, pred.right], dim=-1)
        image = torch.cat([images_gt, images_pred], dim=-2)
        caption = f'psnr={psnr.item()}, mse={mse.item()}, bpp={bpp.item()}, mse_left={mse_left.item()}, mse_right={mse_right.item()}'
        eval_logger.image(image, caption)

    # Log scalars
    eval_logger.scalars(
        ms_ssim=ms_ssim, loss=loss, bpp=bpp, bpp_y=bpp_y, bpp_z=bpp_z, mse=mse, mse_left=mse_left, mse_right=mse_right, psnr=psnr
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""Order of arguments: gpu_idx (required) exp_name (optional).
                                                    The ordering is args > config > resume, meaning for any parameter 
                                                    the cli arguments is used if available, otherwise the specified 
                                                    config file, then the config from the resumed run. 
                                                    If neither is available the default is used.""")
    parser.add_argument('argv', nargs='*', help='gpu_idx (required) exp_name (optional)')
    parser.add_argument('--testing', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--resume', action='store_true', 
                         help='Resume training in same wandb run and experiment folder.')
    parser.add_argument('--resume_new', action='store_true', 
                         help='Resume training in different wandb run and experiment folder.')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--config', type=str)
    parser.add_argument('--project', type=str)
    parser.add_argument('--entity', type=str)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--tags', type=str, 
                         help='A string with tags seperated by commas. E.g.: "tag1, tag2, tag3"')
    parser.add_argument('--lmda', type=float)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr_drop', type=int, 
                         help='number of steps after which the learning rate is dropped.')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--model', type=str, 
                         help=f'Options: {", ".join([l[0] for l in inspect.getmembers(models, inspect.isclass) if l[1].__module__ == "sa.models"])}')
    parser.add_argument('--train', type=str, 
                         help=f'name of training dataset. Options: {", ".join(list_datasets())}')
    parser.add_argument('--eval', type=str, 
                         help=f'name of training dataset. Options: {", ".join(list_datasets())}')
    parser.add_argument('--test', type=str, 
                         help=f'name of training dataset. Options: {", ".join(list_datasets())}')
    args = parser.parse_args()

    config = process_cli_arguments(args)

    # Initialize wandb
    wandb.init(group=config.exp_name, project=config.project, entity=config.entity, 
               tags=config.tags, config=config, id=config.run_id, resume="allow")
    wandb.run.log_code('.')
    wandb.run.name = config.run_id

    experiment(config)
