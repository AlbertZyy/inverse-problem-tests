
from torch.optim.lr_scheduler import _LRScheduler


class MyLR_Scheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps: int, last_epoch=-1, step_power=0.5):
        self.warmup_steps = warmup_steps
        self.step_power = step_power
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step_num = self.last_epoch
        warmup_steps = self.warmup_steps
        sp = self.step_power
        if step_num > 0:
            coef = min(1/step_num**sp, step_num/warmup_steps**(sp+1.))
        else:
            coef = 1./warmup_steps**(sp+1.)
        return [base_lr*coef for base_lr in self.base_lrs]
