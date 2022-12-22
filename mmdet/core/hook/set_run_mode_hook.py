
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class SetRunModeHook(Hook):

    def before_train_epoch(self, runner):
        runner.model.module.run_mode ='train'

    def after_train_epoch(self, runner):
        runner.model.module.run_mode ='test'
