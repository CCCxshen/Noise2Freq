import torch
import torch.nn.functional as F
from torch.autograd import Variable
import lpips
from piq import VIFLoss
from Denoising_indicator import *
from .IQA_indices import nqm
from math import exp

import warnings

warnings.filterwarnings("ignore")


class MetricsCalculator:
    def __init__(self, MIN_HU, MAX_HU, device='cuda'):
        self.MIN_HU = MIN_HU
        self.MAX_HU = MAX_HU
        self.data_range = MAX_HU - MIN_HU
        self.device = device

        self.lpips_fn = lpips.LPIPS(net='vgg').to(self.device)
        # self.lpips_fn = lpips.LPIPS(net='vgg').cuda()
        self.vif_fn = VIFLoss()

    def compute_measure(self, inputs, target, output, is_train=True):
        if not is_train:
            original_lpips = self.lpips_fn(inputs, target).item()
            pred_lpips = self.lpips_fn(output, target).item()

            original_vif = 1 - self.vif_fn(inputs.cpu(), target.cpu()).item()
            pred_vif = 1 - self.vif_fn(output.cpu(), target.cpu()).item()

            original_nqm = nqm(inputs.cpu().numpy().squeeze(), target.cpu().numpy().squeeze())
            pred_nqm = nqm(output.cpu().numpy().squeeze(), target.cpu().numpy().squeeze())
        else:
            original_lpips = pred_lpips = original_vif = pred_vif = original_nqm = pred_nqm = 0.

        inputs = denormalize_(inputs, self.MIN_HU, self.MAX_HU)
        output = denormalize_(output, self.MIN_HU, self.MAX_HU)
        target = denormalize_(target, self.MIN_HU, self.MAX_HU)

        original_psnr = self.compute_PSNR(inputs, target)
        original_ssim = self.compute_SSIM(inputs, target)
        original_rmse = self.compute_RMSE(inputs, target)

        pred_psnr = self.compute_PSNR(output, target)
        pred_ssim = self.compute_SSIM(output, target)
        pred_rmse = self.compute_RMSE(output, target)

        original_results = {
            "rmse": original_rmse,
            "psnr": original_psnr,
            "ssim": original_ssim,
            "lpips": original_lpips,
            "vif": original_vif,
            "nqm": original_nqm,
        }

        pred_results = {
            "rmse": pred_rmse,
            "psnr": pred_psnr,
            "ssim": pred_ssim,
            "lpips": pred_lpips,
            "vif": pred_vif,
            "nqm": pred_nqm,
        }

        return original_results, pred_results

    @staticmethod
    def compute_MSE(img1, img2):
        return ((img1 - img2) ** 2).mean()

    @staticmethod
    def compute_RMSE(img1, img2):
        mse_ = MetricsCalculator.compute_MSE(img1, img2)
        return torch.sqrt(mse_).item()

    def compute_PSNR(self, img1, img2):
        mse_ = self.compute_MSE(img1, img2)
        return 10 * torch.log10((self.data_range ** 2) / mse_).item()

    def compute_SSIM(self, img1, img2, window_size=11, size_average=True):
        channel = img1.size(1)
        window = self.create_window(window_size, channel).type_as(img1)

        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1, C2 = (0.01 * self.data_range) ** 2, (0.03 * self.data_range) ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean().item()
        else:
            return ssim_map.mean(1).mean(1).mean(1).item()

    @staticmethod
    def gaussian(window_size, sigma):
        gauss = torch.Tensor([
            exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)
        ])
        return gauss / gauss.sum()

    @staticmethod
    def create_window(window_size, channel):
        _1D_window = MetricsCalculator.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window
