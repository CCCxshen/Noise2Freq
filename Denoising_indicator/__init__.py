import os
import numpy as np
import matplotlib.pyplot as plt


def normalize_(img, MIN_HU=-1024., MAX_HU=3000.):
    img[img > MAX_HU] = MAX_HU
    img[img < MIN_HU] = MIN_HU
    return (img - MIN_HU) / (MAX_HU - MIN_HU)


def denormalize_(img, MIN_HU=-1024., MAX_HU=3000.):
    img = img * (MAX_HU - MIN_HU) + MIN_HU
    return img


def trans2img(img):
    img = img * 255.
    return np.uint8(img)


def save_slices(
        inputs,
        pred,
        gt,
        out_dir,
        name_prefix,
        epoch,
        idx,
        MIN_HU=-1024.,
        MAX_HU=3000.,
        ori_psnr=0.,
        ori_ssim=0.,
        pred_psnr=0.,
        pred_ssim=0.,
):
    os.makedirs(out_dir, exist_ok=True)

    inputs = denormalize_(inputs.cpu(), MIN_HU, MAX_HU)
    pred = denormalize_(pred.cpu(), MIN_HU, MAX_HU)
    gt = denormalize_(gt.cpu(), MIN_HU, MAX_HU)

    inputs = normalize_(inputs.numpy().squeeze(), -160, 240)
    pred = normalize_(pred.numpy().squeeze(), -160, 240)
    gt = normalize_(gt.numpy().squeeze(), -160, 240)

    plt.imsave(os.path.join(out_dir, f'{name_prefix}_{epoch}_{idx}_input_{ori_psnr}_{ori_ssim}.png'),
               (inputs * 255.).astype(np.uint8),
               cmap='gray')
    plt.imsave(os.path.join(out_dir, f'{name_prefix}_{epoch}_{idx}_pred_{pred_psnr}_{pred_ssim}.png'),
               (pred * 255.).astype(np.uint8),
               cmap='gray')
    plt.imsave(os.path.join(out_dir, f'{name_prefix}_{epoch}_{idx}_gt.png'), (gt * 255.).astype(np.uint8), cmap='gray')
