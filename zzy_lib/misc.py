import builtins as __builtin__
import os
import random
import socket

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import torch
import torch.distributed
from monai.transforms import Randomizable


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
    After setup, can use print(..., force=True) to force printing in non-master processes.
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


def gather_basic_random_states() -> dict:
    """Should be called in all distributed processes."""

    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    states = {
        "torch_rng": torch.get_rng_state(),
        "torch_cuda_rng": torch.cuda.get_rng_state_all(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }

    if torch.distributed.is_initialized():
        gathered_states = [None] * world_size
        torch.distributed.all_gather_object(gathered_states, states)
    else:
        gathered_states = [states]
    gathered_states = {f"rank{i}": gathered_states[i] for i in range(len(gathered_states))}

    return gathered_states


def set_basic_random_states(gathered_states):
    """Should be called in all distributed processes."""

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    states = gathered_states[f"rank{rank}"]

    torch.set_rng_state(states["torch_rng"])
    torch.cuda.set_rng_state_all(states["torch_cuda_rng"])
    np.random.set_state(states["numpy_rng"])
    random.setstate(states["python_rng"])


def gather_monai_transform_random_states(transform) -> dict:
    """Should be called in all distributed processes."""

    def _get(transform, states: dict, prefix: str):
        if isinstance(transform, Randomizable):  # If transform is randomizable, save its state
            states[prefix] = transform.R.get_state()

        # Recursively check attributes of the transform (for nested structures)
        for attr_name in dir(transform):
            if attr_name.startswith("_"):  # Skip private attributes
                continue
            attr = getattr(transform, attr_name)
            if isinstance(attr, Randomizable):  # If the attribute is a randomizable transform
                _get(attr, states, f"{prefix}.{attr_name}")
            elif isinstance(attr, (list, tuple, dict)):  # If it is a container, search inside
                for k, v in attr.items() if isinstance(attr, dict) else enumerate(attr):
                    if isinstance(v, Randomizable):
                        _get(v, states, f"{prefix}.{attr_name}.{k}")

        return states

    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    states = {}
    prefix = ""
    states = _get(transform, states, prefix)

    if torch.distributed.is_initialized():
        gathered_states = [None] * world_size
        torch.distributed.all_gather_object(gathered_states, states)
    else:
        gathered_states = [states]
    gathered_states = {f"rank{i}": gathered_states[i] for i in range(len(gathered_states))}

    return gathered_states


def set_monai_transform_random_states(transform, gathered_states: dict):
    """Should be called in all distributed processes."""

    def _set(transform, states: dict, prefix: str):
        if isinstance(transform, Randomizable):  # If transform is randomizable, restore its state
            transform.set_state(states[prefix])

        # Recursively check attributes of the transform (for nested structures)
        for attr_name in dir(transform):
            if attr_name.startswith("_"):  # Skip private attributes
                continue
            attr = getattr(transform, attr_name)
            if isinstance(attr, Randomizable):  # If the attribute is a randomizable transform
                _set(attr, states, f"{prefix}.{attr_name}")
            elif isinstance(attr, (list, tuple, dict)):  # If it is a container, search inside
                for k, v in attr.items() if isinstance(attr, dict) else enumerate(attr):
                    if isinstance(v, Randomizable) and f"{prefix}.{attr_name}.{k}" in states:
                        _set(v, states, f"{prefix}.{attr_name}.{k}")

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    states = gathered_states[f"rank{rank}"]
    prefix = ""
    _set(transform, states, prefix)


def overlay_segmentation_and_save(image_slice, segmentation_mask, save_path, b_min=0.0, b_max=1.0, alpha=0.3, background_class=0, num_classes=200, seed=42):
    """
    Overlay a segmentation mask on a grayscale image and save as RGB PNG.

    Parameters:
    - image_slice (np.ndarray): 2D grayscale image [H, W]
    - segmentation_mask (np.ndarray): 2D label map [H, W]
    - save_path (str): where to save the PNG
    - b_min, b_max (float): bounds for image normalization
    - alpha (float): blending factor [0,1]
    - background_class (int): label ID to ignore (e.g., background=0)
    - num_classes (int): maximum number of classes to assign colors
    - seed (int): random seed for reproducible colors
    """

    # Normalize grayscale image to [0,1]
    image_norm = (image_slice - b_min) / (b_max - b_min)
    image_norm = np.clip(image_norm, 0, 1)

    # Convert grayscale to RGB
    image_rgb = np.stack([image_norm] * 3, axis=-1)

    # Generate color mapping for classes 0 .. (num_classes-1)
    rng = np.random.default_rng(seed)
    class_colors = {cls: rng.random(3) for cls in range(num_classes)}

    # Create overlay with colors only on masked regions (excluding background)
    overlay = np.zeros_like(image_rgb)
    mask_region = segmentation_mask != background_class
    for cls in range(num_classes):
        if cls == background_class:
            continue  # skip background
        overlay[segmentation_mask == cls] = class_colors[cls]

    # Initialize blended image as grayscale everywhere
    blended = np.copy(image_rgb)

    # Blend only in masked regions
    blended[mask_region] = (1 - alpha) * image_rgb[mask_region] + alpha * overlay[mask_region]

    # Save the result
    plt.imsave(save_path, blended)


def dcm2nii(dcm_dir, nii_path):
    try:
        series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dcm_dir)
        series_reader = sitk.ImageSeriesReader()
        series_reader.SetFileNames(series_file_names)
        image3D = series_reader.Execute()
        sitk.WriteImage(image=image3D, fileName=nii_path, useCompression=True)
    except Exception as e:
        print(f"Error converting {dcm_dir}: {str(e)}")
