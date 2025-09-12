from __future__ import annotations

import collections.abc
from collections.abc import Callable, Sequence
from pathlib import Path

from monai.data import Dataset, PersistentDataset
from monai.data.utils import pickle_hashing
from monai.utils.misc import MAX_SEED
from torch.serialization import DEFAULT_PROTOCOL
from torch.utils.data import ConcatDataset, Subset


def dataset_set_epoch(dataset, epoch):
    if isinstance(dataset, DeterministicDataset) or isinstance(dataset, DeterministicPersistentDataset):
        dataset.set_epoch(epoch)
    elif isinstance(dataset, ConcatDataset):
        for ds in dataset.datasets:
            dataset_set_epoch(ds, epoch)


class DeterministicDataset(Dataset):
    def __init__(self, data: Sequence, transform: Callable | None = None, base_seed=42) -> None:
        super().__init__(data, transform)
        self.base_seed = base_seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        seed = (self.base_seed + index + self.epoch * len(self.data)) % MAX_SEED
        if hasattr(self.transform, "set_random_state"):
            self.transform.set_random_state(seed=seed, state=None)
        # below copied from monai.data.Dataset __getitem__
        if isinstance(index, slice):
            # dataset[:42]
            start, stop, step = index.indices(len(self))
            indices = range(start, stop, step)
            return Subset(dataset=self, indices=indices)
        if isinstance(index, collections.abc.Sequence):
            # dataset[[1, 3, 4]]
            return Subset(dataset=self, indices=index)
        return self._transform(index)


class DeterministicPersistentDataset(PersistentDataset):
    def __init__(
        self,
        data: Sequence,
        transform: Sequence[Callable] | Callable,
        cache_dir: Path | str | None,
        hash_func: Callable[..., bytes] = pickle_hashing,
        pickle_module: str = "pickle",
        pickle_protocol: int = DEFAULT_PROTOCOL,
        hash_transform: Callable[..., bytes] | None = None,
        reset_ops_id: bool = True,
        base_seed=42,
    ) -> None:
        super().__init__(data, transform, cache_dir, hash_func, pickle_module, pickle_protocol, hash_transform, reset_ops_id)
        self.base_seed = base_seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        seed = (self.base_seed + index + self.epoch * len(self.data)) % MAX_SEED
        if hasattr(self.transform, "set_random_state"):
            self.transform.set_random_state(seed=seed, state=None)
        # below copied from monai.data.Dataset __getitem__
        if isinstance(index, slice):
            # dataset[:42]
            start, stop, step = index.indices(len(self))
            indices = range(start, stop, step)
            return Subset(dataset=self, indices=indices)
        if isinstance(index, collections.abc.Sequence):
            # dataset[[1, 3, 4]]
            return Subset(dataset=self, indices=index)
        return self._transform(index)
