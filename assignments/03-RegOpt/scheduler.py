from typing import List

from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    """
    Custom learning rate scheduler class.
    """
    
    def __init__(self, optimizer, last_epoch=-1):
        """
        Create a new scheduler.

        Note to students: You can change the arguments to this constructor,
        if you need to add new parameters.

        """
        # ... Your Code Here ...
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Get the list of learning rates in the schedule.
        """
        # Note to students: You CANNOT change the arguments or return type of
        # this function (because it is called internally by Torch)

        # ... Your Code Here ...
        # accuracy must beat 0.45 on epoch 1 and 0.50 on epoch 2
        lr = 0.001
        decay_rate = 0.5
        lrs = [lr / (1 + i * decay_rate) for i in range(5)]
        # print(lrs)
        return lrs
        # return [i for i in self.base_lrs]
