from math import isclose
from typing import List

import monai.transforms as mtf
import numpy as np
import torch
import torch.nn.functional as F
from monai.config import KeysCollection


def resize_hw_bilinear_then_d_nearest(x, target_hw_sizes, target_d_size):
    """
    x: (C, D, H, W) or (D, H, W)
    returns: (C, target_d_size, target_hw_sizes[0], target_hw_sizes[1])
    """
    if x.ndim == 3:  # (D, H, W) -> (1, D, H, W)
        x = x.unsqueeze(0)

    C, D, H, W = x.shape

    # 1) resize H/W per-slice with bilinear: treat D as "batch"
    y = x.permute(1, 0, 2, 3)  # (D, C, H, W)
    y = F.interpolate(y, size=target_hw_sizes, mode="bilinear", align_corners=False)
    y = y.permute(1, 0, 2, 3)  # (C, D, H_new, W_new)

    # 2) resize D with nearest in 3D (H/W already correct)
    y = y.unsqueeze(0)  # (1, C, D, H_new, W_new)
    y = F.interpolate(y, size=(target_d_size, target_hw_sizes[0], target_hw_sizes[1]), mode="nearest")

    return y.squeeze(0)  # (C, D_new, H_new, W_new)


class MyRandSmoothOrNoised(mtf.MapTransform, mtf.Randomizable):
    """
    Args:
        smooth_kwargs: {"sigma_x": [0.5, 1.2], "sigma_y": [0.5, 1.2], "sigma_z": [0.5, 1.2]}
        noise_kwargs: {"mean": 0.0, "std": 0.1, "sample_std": True}
    """

    def __init__(
        self,
        keys: KeysCollection,
        trans_prob: float,
        smooth_prob: float,
        smooth_kwargs: dict,
        noise_kwargs: dict,
    ) -> None:
        mtf.MapTransform.__init__(self, keys)
        assert trans_prob >= 0 and trans_prob <= 1
        assert smooth_prob >= 0 and smooth_prob <= 1
        self.trans_prob = trans_prob
        self.smooth_prob = smooth_prob
        self.smooth_trans = mtf.RandGaussianSmoothd(keys=keys, prob=1, **smooth_kwargs)
        self.noise_trans = mtf.RandGaussianNoised(keys=keys, prob=1, **noise_kwargs)

    def set_random_state(self, seed: int | None = None, state: np.random.RandomState | None = None):
        super().set_random_state(seed, state)
        self.smooth_trans.set_random_state(seed, state)
        self.noise_trans.set_random_state(seed, state)
        return self

    def __call__(self, data):
        if self.R.rand() < self.trans_prob:
            if self.R.rand() < self.smooth_prob:
                return self.smooth_trans(data)
            else:
                return self.noise_trans(data)
        else:
            return data


class MyRandScaleIntensityRanged(mtf.MapTransform, mtf.Randomizable):
    """
    Args:
        hu_ranges: [[-500, 1000], [-175, 250], [-1000, 2000]]
        probs: [0.4, 0.5, 0.1]
        kwargs: {"b_min": 0, "b_max": 1, "clip": True}
    """

    def __init__(
        self,
        keys: KeysCollection,
        hu_ranges: List[List[float]],
        probs: List[float],
        kwargs: dict,
    ) -> None:
        mtf.MapTransform.__init__(self, keys)
        assert isclose(sum(probs), 1.0)
        self.hu_ranges = hu_ranges
        self.probs = probs
        self.scalers = [mtf.ScaleIntensityRanged(keys=keys, a_min=a[0], a_max=a[1], **kwargs) for a in hu_ranges]

    def set_random_state(self, seed: int | None = None, state: np.random.RandomState | None = None):
        super().set_random_state(seed, state)
        return self

    def __call__(self, data):
        i = self.R.choice(len(self.hu_ranges), p=self.probs)
        return self.scalers[i](data)


class MyMultiWindowScaleIntensityRanged(mtf.MapTransform):
    """
    Args:
        hu_ranges: [[-500, 1000], [-175, 250], [-1000, 2000]]
        has_channel_dim: Assume channel dim is the 0th dim. If False, add a channel dim at the 0th dim.
            For 2 original channels and 3 windows, output will be 2 (window 1) + 2 (window 2) + 2 (window 3) = 6 channels.
        kwargs: {"b_min": 0, "b_max": 1, "clip": True}
    """

    def __init__(
        self,
        keys: KeysCollection,
        hu_ranges: List[List[float]],
        has_channel_dim: bool,
        kwargs: dict,
    ) -> None:
        mtf.MapTransform.__init__(self, keys)
        self.scalers = [mtf.ScaleIntensityRanged(keys=keys, a_min=a[0], a_max=a[1], **kwargs) for a in hu_ranges]
        self.has_channel_dim = has_channel_dim

    def __call__(self, data):
        for k in self.keys:
            outputs = []
            for scaler in self.scalers:
                if self.has_channel_dim:
                    outputs.append(scaler(data)[k])
                else:
                    outputs.append(scaler(data)[k].unsqueeze(0))  # add a channel dim
            data[k] = torch.cat(outputs, dim=0)  # concat along channel dim
        return data


class MyRandSpatialCropWithThresholdOnSourceKeyd(mtf.MapTransform, mtf.Randomizable):
    """
    Args:
        keys: ["image", "label"]
        source_key: "label",
        background_value: 0
        foreground_proportion: 0.4
        max_attempts: 10,
        roi_size: [64, 64, 64]
        random_center: True
        random_size: False
    """

    def __init__(
        self,
        keys: KeysCollection,
        source_key: str,
        background_value: float,
        min_foreground_proportion: float,
        max_attempts: int,
        kwargs: dict,
    ) -> None:
        mtf.MapTransform.__init__(self, keys)
        assert source_key in keys
        assert min_foreground_proportion >= 0 and min_foreground_proportion <= 1
        assert max_attempts >= 1
        self.source_key = source_key
        self.background_value = background_value
        self.foreground_proportion = min_foreground_proportion
        self.max_attempts = max_attempts
        self.cropper = mtf.RandSpatialCropd(keys=keys, **kwargs)

    def set_random_state(self, seed: int | None = None, state: np.random.RandomState | None = None):
        super().set_random_state(seed, state)
        self.cropper.set_random_state(seed, state)
        return self

    def __call__(self, data):
        for _ in range(self.max_attempts):
            cropped_data = self.cropper(data)
            cropped_img = cropped_data[self.source_key]
            fg_prop = np.count_nonzero(cropped_img.numpy() != self.background_value) / np.prod(cropped_img.shape).astype(np.float64)
            if fg_prop >= self.foreground_proportion:
                return cropped_data
        return cropped_data
