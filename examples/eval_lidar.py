import glob
import pytorch_msssim
import tqdm
import argparse
import math
import random
import shutil
import sys
import os
import time
import logging
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.datasets import ImageFolder
from compressai.zoo import image_models

import yaml
import numpy as np
from PIL import Image
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor




class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.lmbda = lmbda
        self.zero = torch.zeros(1).to('cuda')
    
    def psnr(self, output, target):
        mse = torch.mean((output - target) ** 2)
        # if(mse == 0):
        #     return 100
        max_pixel = 1.
        psnr = 10 * torch.log10(max_pixel / mse)
        return torch.mean(psnr)

    def forward(self, output, target, mask=  None, lmbdamap=None):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        if mask is not None:
            mse = self.mse(output['x_hat'], target)
            roi_mse = torch.mean(mse*mask.expand_as(target), [1,2,3])
            # roi_mse_de = torch.sum(mask.repeat(1,3,1,1),[1,2,3])
            out["mse_loss"] = roi_mse.mean()
            roi_mse = torch.mean(roi_mse*(lmbdamap.view(-1,)))
        else:
            out["mse_loss"] = self.mse(output["x_hat"], target)
        out["rdloss"] =  255**2 * roi_mse + out["bpp_loss"]
        
        out["psnr"] = self.psnr(torch.clamp(output["x_hat"],0,1), target)
        return out


class Metrics(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.zero = torch.zeros(1).to('cuda')
    
    def psnr(self, output, target):
        mse = torch.mean((output - target) ** 2, [1,2,3])
        # if(mse == 0):
        #     return 100
        max_pixel = 1.
        psnr = 10 * torch.log10(max_pixel / mse)
        return psnr

    def weighted_psnr(self, img_orig, img_recon, importance_map, max_val=1.0):
        # Ensure shape compatibility
        if importance_map.dim() == 2:
            importance_map = importance_map.unsqueeze(0)  # shape [1, H, W]
        if importance_map.dim() == 4:
            importance_map = importance_map.squeeze(1)

        # Broadcast to all channels
        importance_map = importance_map.expand_as(img_orig)

        mse = ((img_orig - img_recon) ** 2) * importance_map
        weighted_mse = mse.sum() / importance_map.sum()

        psnr = 10 * torch.log10((max_val ** 2) / (weighted_mse + 1e-8))
        return psnr.item()

    def _fspecial_gauss_1d(self, size: int, sigma: float) -> Tensor:
        coords = torch.arange(size, dtype=torch.float)
        coords -= size // 2

        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()

        return g.unsqueeze(0).unsqueeze(0)


    def gaussian_filter(self, input: Tensor, win: Tensor) -> Tensor:
        assert all([ws == 1 for ws in win.shape[1:-1]]), win.shape
        if len(input.shape) == 4:
            conv = F.conv2d
        elif len(input.shape) == 5:
            conv = F.conv3d
        else:
            raise NotImplementedError(input.shape)

        C = input.shape[1]
        out = input
        for i, s in enumerate(input.shape[2:]):
            if s >= win.shape[-1]:
                out = conv(out, weight=win.transpose(2 + i, -1), stride=1, padding=0, groups=C)
            else:
                warnings.warn(
                    f"Skipping Gaussian Smoothing at dimension 2+{i} for input: {input.shape} and win size: {win.shape[-1]}"
                )

        return out


    def my_weighted_ssim(self, X,Y,map,data_range=1.0,K= (0.01, 0.03)):
        for d in range(len(X.shape) - 1, 1, -1):
            X = X.squeeze(dim=d)
            Y = Y.squeeze(dim=d)

        win = self._fspecial_gauss_1d(11, 1.5)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

        K1, K2 = K
        # batch, channel, [depth,] height, width = X.shape
        compensation = 1.0

        C1 = (K1 * data_range) ** 2
        C2 = (K2 * data_range) ** 2

        win = win.to(X.device, dtype=X.dtype)

        mu1 = self.gaussian_filter(X, win)
        mu2 = self.gaussian_filter(Y, win)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = compensation * (self.gaussian_filter(X * X, win) - mu1_sq)
        sigma2_sq = compensation * (self.gaussian_filter(Y * Y, win) - mu2_sq)
        sigma12 = compensation * (self.gaussian_filter(X * Y, win) - mu1_mu2)

        cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)  # set alpha=beta=gamma=1
        ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

        ssim_map = torch.nn.functional.interpolate(ssim_map, size=X.shape[-2:], mode='bilinear', align_corners=False)
        ssim_map = ssim_map.mean(1)
        map = map/map.sum()
        ssim_map = ssim_map * map.unsqueeze(1)
        return ssim_map.sum().item()

    def forward(self, output, target, mask=  None):
        N, _, H, W = target.size()
        out = {}
        num_pixels =  H * W

        bpp = torch.stack([(torch.log(likelihoods).reshape(N,-1).sum(-1) / (-math.log(2) * num_pixels)) for likelihoods in output["likelihoods"].values()])
        bpp = bpp.sum(0)
        out["bpp_loss"] = bpp
        out["mse_loss"] = self.mse(output["x_hat"], target).mean()
        if mask is not None:
            out["roi_psnr"] = torch.mean(10 * torch.log10(1./(torch.sum(((torch.clamp(output["x_hat"],0,1)-target)*mask.repeat(1,3,1,1))**2)/torch.sum(mask.repeat(1,3,1,1)))))
            out['roi_mse'] = torch.mean(((torch.clamp(output["x_hat"],0,1)-target)*mask.repeat(1,3,1,1))**2)
            out["nroi_psnr"] = torch.mean(10 * torch.log10(1./(torch.sum(((torch.clamp(output["x_hat"],0,1)-target)*(1-mask).repeat(1,3,1,1))**2)/torch.sum((1-mask).repeat(1,3,1,1)))))
            out['nroi_mse'] = torch.mean(((torch.clamp(output["x_hat"],0,1)-target)*(1-mask).repeat(1,3,1,1))**2)
            out["roi_psnr_ind"] = 10 * torch.log10(1./(torch.sum(((torch.clamp(output["x_hat"],0,1)-target)*mask.repeat(1,3,1,1))**2, [1,2,3])/torch.sum(mask.repeat(1,3,1,1), [1,2,3])))
            out["nroi_psnr_ind"] = 10 * torch.log10(1./(torch.sum(((torch.clamp(output["x_hat"],0,1)-target)*(1-mask).repeat(1,3,1,1))**2, [1,2,3])/torch.sum((1-mask).repeat(1,3,1,1), [1,2,3])))
            
        out["psnr"] = self.psnr(torch.clamp(output["x_hat"],0,1), target)
        out['ssim'] = pytorch_msssim.ssim(torch.clamp(output["x_hat"],0,1), target, data_range=1.0)
        out["weighted_PSNR"] = self.weighted_psnr(target, torch.clamp(output["x_hat"],0,1), mask)
        out['weighted_ssim'] = self.my_weighted_ssim(target, torch.clamp(output["x_hat"],0,1), mask)
        return out


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def init(args):
    base_dir = f'{args.root}/{args.exp_name}/{args.quality_level}/'
    os.makedirs(base_dir, exist_ok=True)

    return base_dir


def setup_logger(log_dir):
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    log_file_handler = logging.FileHandler(log_dir, encoding='utf-8')
    log_file_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_file_handler)

    log_stream_handler = logging.StreamHandler(sys.stdout)
    log_stream_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_stream_handler)

    logging.info('Logging file is %s' % log_dir)





def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-c",
        "--config",
        default="config/vpt_default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        '--name', 
        default=datetime.now().strftime('%Y-%m-%d_%H_%M_%S'), 
        type=str,
        help='Result dir name', 
    )
    given_configs, remaining = parser.parse_known_args(argv)
    with open(given_configs.config) as file:
        yaml_data= yaml.safe_load(file)
        parser.set_defaults(**yaml_data)
    args = parser.parse_args(remaining)
    return args


class arkitDataset(torch.utils.data.Dataset):
    def __init__(self, root_path, reverse=False, sig=True):
        self.dlist = glob.glob(f'{root_path}/highres_RGB_resized/*')
        self.dlist.sort()
        self.ilist = glob.glob(f'{root_path}/highres_depth_resized/*')
        self.ilist.sort()
        self.dlist = [self.dlist[i] for i in range(len(self.dlist)) if i not in [5529,5530,5531,5532,5533]]
        self.ilist = [self.ilist[i] for i in range(len(self.ilist)) if i not in [5529,5530,5531,5532,5533]]
        self.reverse = reverse
        self.sig = sig


    def __len__(self):
        return len(self.dlist)

    def __getitem__(self, idx):
        image = Image.open(self.dlist[idx]).convert('RGB')
        image = transforms.ToTensor()(image)
        depth = np.array(Image.open(self.ilist[idx]))
        depth_normed = (depth-depth.min())/(depth.max()-depth.min())
        depth_normed_sig = 1 / (1 + np.exp(-15 * (depth_normed - 0.5)))
        if not self.reverse:
            depth_normed = 1 - depth_normed
            depth_normed_sig = 1 - depth_normed_sig
        if self.sig:
            depth = torch.from_numpy(depth_normed_sig).float().unsqueeze(0)
        else:
            depth = torch.from_numpy(depth_normed).float().unsqueeze(0)

        return image, depth
    
def test_all(test_dataloader, model, criterion_rd, metrics, stage='test'):
    model.eval()
    device = next(model.parameters()).device
    lambda_list = [0.0018, 0.0051, 0.013, 0.03665, 0.0932]
    results = {}
    with torch.no_grad():
        for n, lmbda in enumerate(lambda_list):
            loss_am = AverageMeter()
            bpp_loss = AverageMeter()
            mse_loss = AverageMeter()
            aux_loss = AverageMeter()
            psnr = AverageMeter()
            totalloss = AverageMeter()
            weighted_psnr = AverageMeter()
            weighted_msssim = AverageMeter()
            msssim = AverageMeter()
            for i, (d,l) in tqdm.tqdm(enumerate(test_dataloader),leave=False, total=len(test_dataloader.dataset)//test_dataloader.batch_size):
                codecinput = d.to(device)
                roimask = l.to(device)
                lmbda_norm = (np.log(lmbda)-min(np.log(lambda_list)))/(max(np.log(lambda_list))-min(np.log(lambda_list)))
                mask  = torch.zeros(codecinput.shape[0], 1, codecinput.shape[2], codecinput.shape[3], device=device).fill_(lmbda_norm)
                mask_decoder = mask[:,:,:codecinput.shape[2]//16,:codecinput.shape[3]//16]
                lmbda_mask  = torch.zeros(codecinput.shape[0], 1, codecinput.shape[2], codecinput.shape[3], device=device).fill_(lmbda)
                out_net = model(codecinput, mask, mask_decoder, roimask)
                out_criterion = criterion_rd(out_net, codecinput, roimask, torch.tensor([[lmbda]*codecinput.shape[0]]).to(device))
                total_loss = out_criterion['rdloss']

                out_metric = metrics(out_net, codecinput, roimask)

                aux_loss.update(model.aux_loss().item())
                bpp_loss.update(out_criterion["bpp_loss"].item())
                mse_loss.update(out_criterion["mse_loss"].item())
                psnr.update(out_criterion['psnr'].item())
                msssim.update(out_metric['ssim'].item())
                weighted_psnr.update(out_metric['weighted_PSNR'])
                weighted_msssim.update(out_metric['weighted_ssim'])
                totalloss.update(total_loss.item())

            txt = f"{lmbda} || Bpp loss: {bpp_loss.avg:.4f} | weighted_PSNR: {weighted_psnr.avg:.3f}  | PSNR: {psnr.avg:.3f} | Weighted SSIM: {weighted_msssim.avg:.3f} | SSIM: {msssim.avg:.4f}"
            print(txt)
            results[lmbda] = {
                'bpp_loss': bpp_loss.avg,
                'mse_loss': mse_loss.avg,
                'aux_loss': aux_loss.avg,
                'psnr': psnr.avg,
                'weighted_PSNR': weighted_psnr.avg,
                'total_loss': totalloss.avg,
                'ssim': msssim.avg,
                'weighted_ssim': weighted_msssim.avg,
            }
    return results
            
def main(argv):
    args = parse_args(argv)
    base_dir = init(args)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    net = image_models[args.model](quality=int(args.quality_level), prompt_config=args)
    net = net.to(device)
    if args.checkpoint:  # load from previous checkpoint
        logging.info("Loading "+str(args.checkpoint))
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if list(checkpoint["state_dict"].keys())[0][:7]=='module.':
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint["state_dict"].items():
                new_state_dict[k[7:]] = v
        else:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint["state_dict"].items():
                new_state_dict[k] = v
        net.load_state_dict(new_state_dict, strict=True)

    dataset = arkitDataset(args.dataset_path, reverse=False)
    test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    results = test_all(test_dataloader, net, RateDistortionLoss(), Metrics(), stage='test')
    import json
    with open(f'{base_dir}/results_baseline.json', 'w') as f:
        json.dump(results, f)
    logging.info(f"Results saved to {base_dir}/results_baseline.json")

    
if __name__ == "__main__":
    main(sys.argv[1:])