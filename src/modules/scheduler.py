import numpy as np
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingWithWarmUp(_LRScheduler):
        def __init__(
            self,
            optimizer,
            warmup_epoch,
            epoch,
            start_lr,
            warm_lr,
            end_lr,
            last_epoch=-1
        ):
            self.warmup_epoch = warmup_epoch
            self.epoch = epoch
            self.start_lr = start_lr
            self.warm_lr = warm_lr
            self.end_lr = end_lr
            super().__init__(optimizer, last_epoch)

        def get_lr(self):
            if self.warmup_epoch == 0:
                return [(self.start_lr - self.end_lr) * (1 + np.cos(np.pi * self.last_epoch / self.epoch)) / 2 + self.end_lr]
            if self.last_epoch <= self.warmup_epoch:
                return [self.last_epoch * (self.warm_lr - self.start_lr) / self.warmup_epoch + self.start_lr]
            else:
                return [(self.warm_lr - self.end_lr) * (1 + np.cos(np.pi * (self.last_epoch - self.warmup_epoch) / (self.epoch - self.warmup_epoch))) / 2 + self.end_lr]
