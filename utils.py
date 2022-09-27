from typing import Tuple
import torch


def clip_inf_norm(img: torch.Tensor, magnitude: float, std: Tuple[float, float, float]):
    for c in range(3):
        channel_inf_norm = img[c,:,:].detach().abs().max()
        channel_scaled_mag = magnitude / (255.0 * std[c])
        img[c, :, :] = img[c, :, :].clone() * min(1.0, channel_scaled_mag / channel_inf_norm.item())
    return img
