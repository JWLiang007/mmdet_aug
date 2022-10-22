# # Copyright (c) OpenMMLab. All rights reserved.

# from mmcv.runner import HOOKS, Hook
# from mmdet.datasets.pipelines import InstanceAug,RandomCrop,Collect

# @HOOKS.register_module()
# class AutoHp(Hook):

#     def __init__(self,
#                  epoch:int = 12,
#                  by_epoch: bool = True,
#                  **kwargs):
#         self.epoch = epoch
#         self.by_epoch = by_epoch
#         self.args = kwargs

#     def before_epoch(self, runner):
#         if not self.by_epoch:
#             return
#         if runner.epoch == self.epoch:
#             for loss_name, loss_param in runner.model.module.distill_losses.items():
#                 if loss_name.startswith('adv'):
#                     loss_param.alpha_adv=0.0
#             transforms = runner.data_loader.sampler.dataset.pipeline.transforms
#             for transform in transforms:
#                 if isinstance(transform, InstanceAug) or isinstance(transform, RandomCrop):
#                     transforms.pop(transform)
#                 if isinstance(transform, Collect):
#                     if 'adv' in transform.keys:
#                         transform.keys.pop('adv')

#         pass


