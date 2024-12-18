import math 
from torch.optim.lr_scheduler import _LRScheduler 


class WarmupExponentialLR(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, gamma, last_epoch=-1):
        
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        super(WarmupExponentialLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            warmup_factor = (self.last_epoch + 1) / float(self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        
        # ExponentialLR 
        decay_step = self.last_epoch - self.warmup_steps
        return [base_lr * (self.gamma ** decay_step) for base_lr in self.base_lrs]



class MultiStageLRScheduler(_LRScheduler):
    def __init__(self, optimizer, milestones, factors, last_epoch=-1):
        """
        Multi-stage learning rate scheduler.
        :param optimizer: Optimizer
        :param milestones: List of epochs where learning rate changes
        :param factors: List of scaling factors for each milestone
        """
        
        self.milestones = milestones
        self.factors = factors
        assert len(milestones) == len(factors), "Milestones and factors must have the same length."
        super(MultiStageLRScheduler, self).__init__(optimizer, last_epoch)
        

    def get_lr(self):
        # factor accoring to the current learning rate
        factor = 1.0
        for milestone, f in zip(self.milestones, self.factors):
            if self.last_epoch >= milestone:
                factor = f
            else:
                break

        # return the learning rate after modification
        return [base_lr * factor for base_lr in self.base_lrs]



class WarmupCosineLRScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        super(WarmupCosineLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Warm-up stage：learning rate growing
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_epochs
                for base_lr in self.base_lrs
            ]
        else:
            # cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            return [
                base_lr * 0.5 * (1 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]