import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
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
from utils.util import *
from dataset.patDataset import *
from Denoising_indicator import *
from utils.matplotlib_util import *
from dataset.get_dataset_info import get_data_info

def T2N(a):
    if len(a.shape) == 4:
        return a[0][0].detach().cpu().numpy()
    elif len(a.shape) == 3:
        return a[0].detach().cpu().numpy()
    elif len(a.shape) == 2:
        return a.detach().cpu().numpy()

# Modify the second item to change the test dataset, and modify the third item to alter the training mode of the model.
reference_version = "[best_model]-[dataset_name]-[model_mode]"

def parse_args():

    parser = argparse.ArgumentParser(description="Distributed Training Configuration")
    parser.add_argument("--dataset_name", default="mayo2016_3mm", type=str, choices=["siemens", "mayo", "mayo_1mm", "mvct_kvct"])
    parser.add_argument("--encoder_type", default="MCCM", type=str, choices=["norm", "MCCM"])
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--save_dir", default=f"results/{reference_version}/checkpoints/", type=str)
    parser.add_argument("--log_dir", default=f"inference/{reference_version}/", type=str)
    parser.add_argument("--out_dir", default=f"inference/{reference_version}/out/", type=str)
    parser.add_argument("--npy_dir", default=f"inference/{reference_version}/npy/", type=str)
    parser.add_argument("--task_dir", default=f"results_medical/{reference_version}/", type=str)
    parser.add_argument("--gpu", default=1, type=float)
    
    return parser.parse_args()

def test():
    set_seed(1234)
    args = parse_args()
    args.gpu = os.environ["CUDA_VISIBLE_DEVICES"]
    device = torch.device(f"cuda")

    args.dataset_name = reference_version.split(']')[1].split('[')[1]
    
    print(args)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.npy_dir, exist_ok=True)

    args.data_dir, _, _, test_ids = get_data_info(args.dataset_name)
    
    with open(os.path.join(args.log_dir, "indicator.txt"), "a") as f:
        print("--------------------------------------------------------------")
        print("CUDA_VISIBLE_DEVICES is ", os.environ["CUDA_VISIBLE_DEVICES"])
        print(f"test version: {reference_version}")
        f.write(f"test version: {reference_version}\n") 
        
        args_dict = vars(args)
        for key, value in args_dict.items():
            print(f"{key}: {value}")
            f.write(f"{key}: {value}\n")  
        print("--------------------------------------------------------------")  

    # test_dataset = PairedCTDataset(
    #     data_dir=args.data_dir,
    #     patient_ids=test_ids,
    #     load_mode=1
    # )
    test_dataset = patDataset(
        data_dir=args.data_dir,
        is_train = False,
        regex_pattern=[
            rf"({'|'.join(test_ids)})_[a-zA-Z0-9]+_input.npy",
            rf"({'|'.join(test_ids)})_[a-zA-Z0-9]+_target.npy"
        ]
    )

    print(len(test_dataset))

    val_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    TE = ender(1, 1, ema = True, encoder_type=args.encoder_type).to(device)
    TD = deder(1, 1, ema = True, decoder_type=args.encoder_type).to(device)
    
    # dict_path = os.path.join(args.save_dir, "best_modelT.pth")
    dict_path = os.path.join("/data/xcshen/research/TP1_Denoising/Noise2Freq/best_ckpt/siemens_small/NDCT/best_modelT.pth")
    # dict_path = "/data0/xcshen/research/xcshen_research_topic/TP1_Noise2Freq/Noise2Freq_explore/best_model_CZSS/mvct_kvct/NDCT/best_modelT.pth"
    state_dict = torch.load(dict_path)
    print(state_dict.keys())
    TE.load_state_dict(state_dict["TE"])
    TD.load_state_dict(state_dict["TD"])
    print(f"loading from {dict_path}")

    metric_calc = MetricsCalculator(MIN_HU=-1024, MAX_HU=1000, device=device)
    
    indicator = test_date((TE, TD), val_loader, metric_calc, device, args)
    
    indicator_txt = f" rmse:{indicator[0]:.4f}\n psnr:{indicator[1]:.4f}\n ssim:{indicator[2]:.4f}\nlpips:{indicator[3]:.4f}\n  vif:{indicator[4]:.4f}\n  nqm:{indicator[5]:.4f}"
    with open(os.path.join(args.log_dir, "indicator.txt"), 'a') as f:
        f.write(f"{reference_version}\n\n")
        f.write(indicator_txt)
    print(f"{reference_version}\n")
    print(f"{indicator_txt}")


@torch.no_grad()
def test_date(T, val_loader, metric_calc, device, args):
    T[0].eval()
    T[1].eval()
    
    rmse_sum_T = 0.0
    psnr_sum_T = 0.0
    ssim_sum_T = 0.0
    lpips_sum_T = 0.0
    vif_sum_T = 0.0
    nqm_sum_T = 0.0
    
    n_samples = 0

    indicator_record = [[],[],[],[],[],[]]
    
    for batch in tqdm(val_loader, desc="[Validation]", leave=False):
        x = batch["input"].float().to(device).unsqueeze(0)  # [1, 1, H, W]
        y = batch["target"].float().to(device).unsqueeze(0)  # [1, 1, H, W]

        
        skips, x_TE = T[0](x)
        pred_TD = T[1](x, skips, x_TE)
        
        pred_TD = torch.clamp(pred_TD, 0.0, 1.0)

        _, pred_TD_res = metric_calc.compute_measure(x, y, pred_TD, is_train=False)
    
        if n_samples % 1 == 0:
            pred_TD = denormalize_(pred_TD, -1024, 1000)
            x = denormalize_(x, -1024, 1000)
            y = denormalize_(y, -1024, 1000)
            show_2Dimages(
                images = [
                    T2N(x), T2N(y), T2N(pred_TD),
                    T2N(normalize_(torch.clamp(x, -160, 240), -160, 240) * 255),
                    T2N(normalize_(torch.clamp(y, -160, 240), -160, 240) * 255),
                    T2N(normalize_(torch.clamp(pred_TD, -160, 240), -160, 240) * 255),
                ],
                names = [
                    "x", "y","pred", 
                    "x[-160, 240]", "y[-160, 240]", "pred[-160, 240]"
                ],
                shape = (2, 3),
                title = f"{reference_version}-INFERENCE\nval step={n_samples} \nrmse={pred_TD_res['rmse']:.4f}, psnr={pred_TD_res['psnr']:.4f}, ssim={pred_TD_res['ssim']:.4f}, lpips={pred_TD_res['lpips']:.4f}, vif={pred_TD_res['vif']:.4f}, nqm={pred_TD_res['nqm']:.4f}",
                save_name = f"step={n_samples}",
                save_path = args.out_dir
            )

            np.save(os.path.join(args.npy_dir, f"step_{n_samples}_target"), T2N(y))
            np.save(os.path.join(args.npy_dir, f"step_{n_samples}_pred"), T2N(pred_TD))
            np.save(os.path.join(args.npy_dir, f"step_{n_samples}_input"), T2N(x))
            
        rmse_sum_T += pred_TD_res['rmse']
        psnr_sum_T += pred_TD_res['psnr']
        ssim_sum_T += pred_TD_res['ssim']
        lpips_sum_T += pred_TD_res['lpips']
        vif_sum_T += pred_TD_res['vif']
        nqm_sum_T += pred_TD_res['nqm']
        
        indicator_record[0].append(pred_TD_res["rmse"])
        indicator_record[1].append(pred_TD_res["psnr"])
        indicator_record[2].append(pred_TD_res["ssim"])
        indicator_record[3].append(pred_TD_res["lpips"])
        indicator_record[4].append(pred_TD_res["vif"])
        indicator_record[5].append(pred_TD_res["nqm"])
        
        n_samples += 1

    avg_rmse_T = rmse_sum_T / n_samples if n_samples > 0 else 0.0
    avg_psnr_T = psnr_sum_T / n_samples if n_samples > 0 else 0.0
    avg_ssim_T = ssim_sum_T / n_samples if n_samples > 0 else 0.0
    avg_lpips_T = lpips_sum_T / n_samples if n_samples > 0 else 0.0
    avg_vif_T = vif_sum_T / n_samples if n_samples > 0 else 0.0
    avg_nqm_T = nqm_sum_T / n_samples if n_samples > 0 else 0.0
    
    np.save(os.path.join(args.npy_dir, f"indicator"), np.array(indicator_record))    

    return (avg_rmse_T, avg_psnr_T, avg_ssim_T, avg_lpips_T, avg_vif_T, avg_nqm_T)

if __name__ == "__main__":
    test()
