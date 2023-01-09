
from mmcv.runner import HOOKS, Hook
import numpy as np 


@HOOKS.register_module()
class SetLossWeightHook(Hook):

    def __init__(self, start_epoch=1,eta_min=0,eta_max=1,attrs= None ):
        self.start_epoch = start_epoch 
        self.eta_min=eta_min
        self.eta_max = eta_max
        self.attrs = attrs
        
        self.losses  = None 
        self.ori_val = {}
        
    def before_train_epoch(self, runner):

        cur_epoch = runner.epoch + 1
        if cur_epoch <  self.start_epoch:
            return 
        
        if self.losses is None :
            self.losses = dict(runner.model.module.distill_losses)
        coefficient = self.eta_min + 0.5 * (self.eta_max - self.eta_min) * (1 + np.cos(np.pi * (cur_epoch - self.start_epoch) / runner.max_epochs))
    
        for loss_name, attrs in self.attrs.items():
            assert loss_name in self.losses
            for attr in attrs:
                loss_obj = self.losses[loss_name]
                assert hasattr(loss_obj,attr)
                _attr = '_'.join([loss_name,attr])
                if _attr not in self.ori_val:
                    self.ori_val[_attr] = getattr(loss_obj,attr)
                new_val = coefficient * self.ori_val[_attr]
                setattr(loss_obj,attr,new_val)
                pass


    def after_train_epoch(self, runner):

        return 
