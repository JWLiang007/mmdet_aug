# Copyright (c) OpenMMLab. All rights reserved.
from pathlib import Path
import torch.nn as nn
import mmcv
import torch
from mmcv.runner import load_checkpoint

from mmcv.runner import  load_checkpoint, _load_checkpoint, load_state_dict
from mmdet.distillation.builder import DISTILLER,build_distill_loss
from collections import OrderedDict

from .. import build_detector
from ..builder import DETECTORS
from .single_stage import SingleStageDetector


@DETECTORS.register_module()
class KnowledgeDistillationSingleStageDetector(SingleStageDetector):
    r"""Implementation of `Distilling the Knowledge in a Neural Network.
    <https://arxiv.org/abs/1503.02531>`_.

    Args:
        teacher_config (str | dict): Config file path
            or the config object of teacher model.
        teacher_ckpt (str, optional): Checkpoint path of teacher model.
            If left as None, the model will not load any weights.
    """

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 teacher_config,
                 distill_cfg=None,
                 teacher_ckpt=None,
                 eval_teacher=True,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__(backbone, neck, bbox_head, train_cfg, test_cfg,
                         pretrained)
        self.eval_teacher = eval_teacher
        # Build teacher model
        if isinstance(teacher_config, (str, Path)):
            teacher_config = mmcv.Config.fromfile(teacher_config)
        self.teacher_model = build_detector(teacher_config['model'])
        if teacher_ckpt is not None:
            load_checkpoint(
                self.teacher_model, teacher_ckpt, map_location='cpu')
        if distill_cfg is not None:
            self.distill_losses = nn.ModuleDict()
            self.distill_cfg = distill_cfg

            student_modules = dict(self.named_modules())
            teacher_modules = dict(self.teacher_model.named_modules())

            def regitster_hooks(student_module, teacher_module):
                def hook_teacher_forward(module, input, output):
                    self.register_buffer(teacher_module, output)

                def hook_student_forward(module, input, output):
                    self.register_buffer(student_module, output)

                return hook_teacher_forward, hook_student_forward

            for item_loc in self.distill_cfg:

                student_module = 'student_' + item_loc.student_module.replace('.', '_')
                teacher_module = 'teacher_' + item_loc.teacher_module.replace('.', '_')

                self.register_buffer(student_module, None)
                self.register_buffer(teacher_module, None)

                hook_teacher_forward, hook_student_forward = regitster_hooks(student_module, teacher_module)
                teacher_modules[item_loc.teacher_module].register_forward_hook(hook_teacher_forward)
                student_modules[item_loc.student_module].register_forward_hook(hook_student_forward)

                for item_loss in item_loc.methods:
                    loss_name = item_loss.name
                    self.distill_losses[loss_name] = build_distill_loss(item_loss)
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        adv_feature_loss = {}
        if 'adv' in kwargs.keys():
            adv_img = kwargs.pop('adv')
            adv_feat_s = self.extract_feat(adv_img)
            with torch.no_grad():
                self.teacher_model.eval()
                adv_feat_t = self.teacher_model.extract_feat(adv_img)
            buffer_dict = dict(self.named_buffers())
            for item_loc in self.distill_cfg:

                student_module = 'student_' + item_loc.student_module.replace('.', '_')
                teacher_module = 'teacher_' + item_loc.teacher_module.replace('.', '_')

                student_feat = buffer_dict[student_module]
                teacher_feat = buffer_dict[teacher_module]

                for item_loss in item_loc.methods:
                    loss_name = item_loss.name
                    if str(loss_name).startswith('adv'):
                        adv_feature_loss[loss_name] = self.distill_losses[loss_name](adv_feat_s, adv_feat_t)

                    else:
                        adv_feature_loss[loss_name] = self.distill_losses[loss_name](student_feat, teacher_feat)

        x = self.extract_feat(img)
        with torch.no_grad():
            teacher_x = self.teacher_model.extract_feat(img)
            out_teacher = self.teacher_model.bbox_head(teacher_x)
        losses = self.bbox_head.forward_train(x, out_teacher, img_metas,
                                              gt_bboxes, gt_labels,
                                              gt_bboxes_ignore)

        losses.update(adv_feature_loss)
        return losses

    def cuda(self, device=None):
        """Since teacher_model is registered as a plain object, it is necessary
        to put the teacher model to cuda when calling cuda function."""
        self.teacher_model.cuda(device=device)
        return super().cuda(device=device)

    def train(self, mode=True):
        """Set the same train mode for teacher and student model."""
        if self.eval_teacher:
            self.teacher_model.train(False)
        else:
            self.teacher_model.train(mode)
        super().train(mode)

    def __setattr__(self, name, value):
        """Set attribute, i.e. self.name = value

        This reloading prevent the teacher model from being registered as a
        nn.Module. The teacher module is registered as a plain object, so that
        the teacher parameters will not show up when calling
        ``self.parameters``, ``self.modules``, ``self.children`` methods.
        """
        if name == 'teacher_model':
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)
