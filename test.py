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

def experiment(CFG, model_path=None):
    """
    Training initialization and loop.
    """

    # Init dataloader
    test_set = Dataset(data_type='test', **CFG.test, pos_encoding=CFG.model.kwargs.get('pos_encoding', False))
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False,
                              num_workers=4, pin_memory=False)
    test_logger = Logger(CFG.test.name + '#test')

    # initialize model
    model = getattr(models, CFG.model.name)(**CFG.model.kwargs)
    model = model.to(CFG.device)

    checkpoint_model = torch.load(model_path, map_location=CFG.device)
    model.load_state_dict(checkpoint_model)

    # test model after training
    print(f'\n## TESTING ON {test_logger.prefix} ##')
    with torch.no_grad():
        evaluation(test_loader, model, test_logger, CFG)
        test_results = test_logger.scal.copy()
        test_logger.log(0)
        
    return test_results


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
    context_warping_loss = latents.context_warping_loss
    warping_loss = latents.he_warping_loss + latents.hd_warping_loss + latents.d_warping_loss + latents.e_warping_loss
    loss = (bpp + CFG.lmda * mse) / (1 + CFG.lmda)

    if CFG.log_images:
        images_gt = torch.cat([left, right], dim=-1)
        images_pred = torch.cat([pred.left, pred.right], dim=-1)
        image = torch.cat([images_gt, images_pred], dim=-2)
        caption = f'psnr={psnr.item()}, mse={mse.item()}, bpp={bpp.item()}, mse_left={mse_left.item()}, mse_right={mse_right.item()}'
        eval_logger.image(image, caption)

    # Log scalars
    eval_logger.scalars(
        he_warping_loss=latents.he_warping_loss, hd_warping_loss=latents.hd_warping_loss, e_warping_loss=latents.e_warping_loss, d_warping_loss=latents.d_warping_loss, ms_ssim=ms_ssim,
        context_warping_loss=context_warping_loss, warping_loss=warping_loss, loss=loss, bpp=bpp, bpp_y=bpp_y, bpp_z=bpp_z, mse=mse, mse_left=mse_left, mse_right=mse_right, psnr=psnr
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""Positional argument: gpu_idx (required) exp_name (optional)""")
    parser.add_argument('argv', nargs='*', help='gpu_idx (required) exp_name (optional)')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--tags', type=str, 
                         help='A string with tags seperated by commas. E.g.: "tag1, tag2, tag3"')
    parser.add_argument('--resume_dir', type=str, help='directory that contains config.json and model.pt') 
    args = parser.parse_args()

    args.config = args.resume_dir + '/config.json'
    model_path = args.resume_dir + '/model.pt'

    config = process_cli_arguments(args)

    # Initialize wandb
    wandb.init(group=config.exp_name, project=config.project, entity=config.entity, 
               tags=config.tags, config=config, id=config.run_id, resume="allow")
    wandb.run.log_code('.')
    wandb.run.name = config.run_id

    experiment(config, model_path=model_path)
