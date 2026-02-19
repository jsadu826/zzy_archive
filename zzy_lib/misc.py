import builtins as __builtin__
import random
import socket

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

    def synchronize_between_processes(self):
        if not torch.distributed.is_initialized():
            return self.avg
        cnt = torch.tensor(self.cnt).cuda()
        torch.distributed.all_reduce(cnt, torch.distributed.ReduceOp.SUM)
        sum = torch.tensor(self.sum).cuda()
        torch.distributed.all_reduce(sum, torch.distributed.ReduceOp.SUM)
        avg = sum / cnt
        return avg.cpu().item()


class MultiLossAverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.meters = {}

    def update(self, losses_dict):
        """
        losses_dict (dict): {'name1': (val1, n1), 'name2': (val2, n2), ...}
        """
        for loss_name, (val, n) in losses_dict.items():
            if loss_name not in self.meters:
                self.meters[loss_name] = AverageMeter()
            self.meters[loss_name].update(val, n)

    def synchronize_between_processes(self):
        return {loss_name: meter.synchronize_between_processes() for loss_name, meter in self.meters.items()}


def print_func_setup(is_master):
    """
    This function disables printing when not in master process.
    After setup, can use `print(..., force=True)` to force printing in non-master processes.
    """
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs, flush=True)

    __builtin__.print = print


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "y", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "n", "false", "f", "0"):
        return False
    else:
        raise ValueError(f"Unsupported str `{v}`!")


def int_with_none(v):
    if v is None:
        return None
    else:
        return int(v)


def find_available_port(start_port=20000, end_port=30000):
    for port in range(start_port, end_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) == 1:
                return port
    print(f"No available port found in range [{start_port}, {end_port}]!")
    return None


def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def overlay_seg_and_save(img, seg, save_path, b_min=0.0, b_max=1.0, alpha=0.3, background_class=0, num_classes=200, seed=42):
    """
    Overlay a segmentation mask on a grayscale image and save as RGB PNG.
    Args:
        img (np.ndarray): 2D grayscale image [H, W]
        seg (np.ndarray): 2D label image [H, W]
        save_path (str): where to save the PNG
        b_min (float): lower bound for intensity normalization
        b_max (float): upper bound for intensity normalization
        alpha (float): blending factor [0,1]
        background_class (int): label ID to ignore (e.g., background=0)
        num_classes (int): maximum number of classes to assign colors
        seed (int): random seed for reproducible colors
    """

    # Normalize grayscale image to [0,1]
    normed_img = (img - b_min) / (b_max - b_min)
    normed_img = np.clip(normed_img, 0, 1)

    # Convert grayscale to RGB
    rgb_img = np.stack([normed_img] * 3, axis=-1)

    # Generate color mapping for classes 0 .. (num_classes-1)
    rng = np.random.default_rng(seed)
    class_colors = {cls: rng.random(3) for cls in range(num_classes)}

    # Create overlay with colors only on masked regions (excluding background)
    overlay_img = np.zeros_like(rgb_img)
    seg_region = seg != background_class
    for cls in range(num_classes):
        if cls == background_class:
            continue  # skip background
        overlay_img[seg == cls] = class_colors[cls]

    # Initialize blended image as grayscale everywhere
    blended_img = np.copy(rgb_img)

    # Blend only in masked regions
    blended_img[seg_region] = (1 - alpha) * rgb_img[seg_region] + alpha * overlay_img[seg_region]

    # Save the result
    plt.imsave(save_path, blended_img)
    print("Overlaid image saved to:", save_path)
