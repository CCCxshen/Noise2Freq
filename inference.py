import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from networks.net_factory import * 
from utils.loss import MSELoss, L1Loss
from torch.optim import Adam, AdamW
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from Denoising_indicator.metrics import MetricsCalculator
from utils.util import set_seed
from torch.utils.tensorboard.writer import SummaryWriter
import torch.nn as nn
import argparse
import torch.nn.functional as F
from datasets import *
from utils.util import *
from Denoising_indicator import *
from utils.matplotlib_util import *

def T2N(a):
    if len(a.shape) == 4:
        return a[0][0].detach().cpu().numpy()
    elif len(a.shape) == 3:
        return a[0].detach().cpu().numpy()
    elif len(a.shape) == 2:
        return a.detach().cpu().numpy()

reference_version = "mayo_ndct"
def parse_args():

    parser = argparse.ArgumentParser(description="Distributed Training Configuration")
    parser.add_argument("--dataset_name", default="mayo", type=str)
    parser.add_argument(
        "--data_dir", default="/data/fhzhang/data/mayo_processed_data", type=str
    )
    # parser.add_argument(
    #     "--data_dir", default="/data0/xcshen/data/siemens_processed_data", type=str
    # )
    parser.add_argument(
        "--save_dir", default=f"results/{reference_version}/checkpoints/", type=str
    )
    parser.add_argument("--log_dir", default=f"inference/{reference_version}/", type=str)
    parser.add_argument("--out_dir", default=f"inference/{reference_version}/out/", type=str)
    parser.add_argument("--task_dir", default=f"results_medical/{reference_version}/", type=str)
    parser.add_argument("--gpu", default=3, type=float)
    
    return parser.parse_args()


def test():
    set_seed(1234)
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device(f"cuda")

    print(args)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.dataset_name == "mayo":
        test_ids = ["L333", "L506"]

    test_dataset = PairedCTDataset(
        data_dir=args.data_dir,
        patient_ids=test_ids,
        load_mode=1
    )

    print(len(test_dataset))

    val_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

   
    EB = ender(1, 1, ema = True).to(device)
    DB = deder(1, 1, ema = True).to(device)
    
    dict_path = os.path.join(args.save_dir, "best_modelB.pth")
    state_dict = torch.load(dict_path)
    EB.load_state_dict(state_dict["EB"])
    DB.load_state_dict(state_dict["DB"])
    print(f"loading from {dict_path}")

    metric_calc = MetricsCalculator(MIN_HU=-1024, MAX_HU=1000, device=device)
    
    indicator = test_date((EB, DB), val_loader, metric_calc, device, args)
    
    indicator_txt = f" rmse:{indicator[0]:.4f}\n psnr:{indicator[1]:.4f}\n ssim:{indicator[2]:.4f}\nlpips:{indicator[3]:.4f}\n  vif:{indicator[4]:.4f}\n  nqm:{indicator[5]:.4f}"
    with open(os.path.join(args.log_dir, "indicator.txt"), 'a') as f:
        f.write(f"{reference_version}\n\n")
        f.write(indicator_txt)
    print(f"{reference_version}\n")
    print(f"{indicator_txt}")


@torch.no_grad()
def test_date(B, val_loader, metric_calc, device, args):
    B[0].eval()
    B[1].eval()
    
    
    rmse_sum_B = 0.0
    psnr_sum_B = 0.0
    ssim_sum_B = 0.0
    lpips_sum_B = 0.0
    vif_sum_B = 0.0
    nqm_sum_B = 0.0
    
    n_samples = 0

    for batch in tqdm(val_loader, desc="[Validation]", leave=False):
        x = batch["input"].float().to(device).unsqueeze(0)  # [1, 1, H, W]
        y = batch["target"].float().to(device).unsqueeze(0)  # [1, 1, H, W]

        
        skips, x_EB = B[0](x)
        pred_DB = B[1](x, skips, x_EB)
        
        pred_DB = torch.clamp(pred_DB, 0.0, 1.0)

        _, pred_DB_res = metric_calc.compute_measure(x, y, pred_DB, is_train=False)
    
        if n_samples % 5 == 0:
            pred_DB = denormalize_(pred_DB, -1024, 1000)
            x = denormalize_(x, -1024, 1000)
            y = denormalize_(y, -1024, 1000)
            show_2Dimages(
                images = [
                    T2N(x), T2N(y), T2N(pred_DB),
                    T2N(normalize_(torch.clamp(x, -160, 240), -160, 240) * 255),
                    T2N(normalize_(torch.clamp(y, -160, 240), -160, 240) * 255),
                    T2N(normalize_(torch.clamp(pred_DB, -160, 240), -160, 240) * 255),
                ],
                names = [
                    "x", "y","pred_B", 
                    "x[-160, 240]", "y[-160, 240]", "pred_B[-160, 240]"
                ],
                shape = (2, 3),
                title = f"val step={n_samples} \nrmse={pred_DB_res['rmse']:.4f}, psnr={pred_DB_res['psnr']:.4f}, ssim={pred_DB_res['ssim']:.4f}, lpips={pred_DB_res['lpips']:.4f}, vif={pred_DB_res['vif']:.4f}, nqm={pred_DB_res['nqm']:.4f}",
                save_name = f"step={n_samples}",
                save_path = args.out_dir
            )
        
        rmse_sum_B += pred_DB_res['rmse']
        psnr_sum_B += pred_DB_res['psnr']
        ssim_sum_B += pred_DB_res['ssim']
        lpips_sum_B += pred_DB_res['lpips']
        vif_sum_B += pred_DB_res['vif']
        nqm_sum_B += pred_DB_res['nqm']
        
        n_samples += 1

    avg_rmse_B = rmse_sum_B / n_samples if n_samples > 0 else 0.0
    avg_psnr_B = psnr_sum_B / n_samples if n_samples > 0 else 0.0
    avg_ssim_B = ssim_sum_B / n_samples if n_samples > 0 else 0.0
    avg_lpips_B = lpips_sum_B / n_samples if n_samples > 0 else 0.0
    avg_vif_B = vif_sum_B / n_samples if n_samples > 0 else 0.0
    avg_nqm_B = nqm_sum_B / n_samples if n_samples > 0 else 0.0
    

    return (avg_rmse_B, avg_psnr_B, avg_ssim_B, avg_lpips_B, avg_vif_B, avg_nqm_B)

if __name__ == "__main__":
    test()
