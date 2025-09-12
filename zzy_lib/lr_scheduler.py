import math
import warnings

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """warmup_epochs = 0: no warmup"""

    def __init__(
        self,
        optimizer: Optimizer,
        max_epochs: int,
        warmup_epochs: int = 0,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:

        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, please use `get_last_lr()`.", UserWarning)

        if self.last_epoch < self.warmup_epochs:
            return [self.warmup_start_lr + (base_lr - self.warmup_start_lr) * self.last_epoch / self.warmup_epochs for base_lr in self.base_lrs]
        elif self.last_epoch <= self.max_epochs:
            return [self.eta_min + 0.5 * (base_lr - self.eta_min) * (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs))) for base_lr in self.base_lrs]
        else:
            return [self.eta_min] * len(self.base_lrs)

    def _get_closed_form_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [self.warmup_start_lr + (base_lr - self.warmup_start_lr) * self.last_epoch / self.warmup_epochs for base_lr in self.base_lrs]
        elif self.last_epoch <= self.max_epochs:
            return [self.eta_min + 0.5 * (base_lr - self.eta_min) * (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs))) for base_lr in self.base_lrs]
        else:
            return [self.eta_min] * len(self.base_lrs)


class LinearWarmupConstantLR(_LRScheduler):
    """warmup_epochs = 0: no warmup"""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int = 0,
        warmup_start_lr: float = 0.0,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:

        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, please use `get_last_lr()`.", UserWarning)

        if self.last_epoch < self.warmup_epochs:
            return [self.warmup_start_lr + (base_lr - self.warmup_start_lr) * self.last_epoch / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            return self.base_lrs

    def _get_closed_form_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [self.warmup_start_lr + (base_lr - self.warmup_start_lr) * self.last_epoch / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            return self.base_lrs


if __name__ == "__main__":
    import torch

    model = torch.nn.Linear(50, 50)
    optim_cosine = torch.optim.AdamW(params=model.parameters(), lr=1.0, amsgrad=True)
    optim_constant = torch.optim.AdamW(params=model.parameters(), lr=1.0, amsgrad=True)
    cosine_sched = LinearWarmupCosineAnnealingLR(optimizer=optim_cosine, warmup_epochs=10, warmup_start_lr=0.233, max_epochs=30, eta_min=1e-3)
    constant_sched = LinearWarmupConstantLR(optimizer=optim_constant, warmup_epochs=10, warmup_start_lr=0.233)
    for i in range(40):
        print(f'Before step {i+1:<12} cosine: {optim_cosine.param_groups[0]["lr"]:<20,.8f} constant: {optim_constant.param_groups[0]["lr"]:.8f}')
        optim_cosine.step(), optim_constant.step()
        cosine_sched.step(), constant_sched.step()
