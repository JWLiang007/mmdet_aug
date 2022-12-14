import torch.nn as nn
import torch
from mmdet.models.detectors.base import BaseDetector
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint, _load_checkpoint, load_state_dict
from ..builder import DISTILLER, build_distill_loss
from collections import OrderedDict


@DISTILLER.register_module()
class FGDDistiller(BaseDetector):
    """Base distiller for detectors.

    It typically consists of teacher_model and student_model.
    """

    def __init__(
        self,
        teacher_cfg,
        student_cfg,
        distill_cfg=None,
        teacher_pretrained=None,
        init_student=False,
    ):

        super(FGDDistiller, self).__init__()

        self.teacher = build_detector(
            teacher_cfg.model,
            train_cfg=teacher_cfg.get("train_cfg"),
            test_cfg=teacher_cfg.get("test_cfg"),
        )
        self.init_weights_teacher(teacher_pretrained)
        self.teacher.eval()

        self.student = build_detector(
            student_cfg.model,
            train_cfg=student_cfg.get("train_cfg"),
            test_cfg=student_cfg.get("test_cfg"),
        )
        self.student.init_weights()
        if init_student:
            t_checkpoint = _load_checkpoint(teacher_pretrained)
            all_name = []
            for name, v in t_checkpoint["state_dict"].items():
                if name.startswith("backbone."):
                    continue
                else:
                    all_name.append((name, v))

            state_dict = OrderedDict(all_name)
            load_state_dict(self.student, state_dict)

        self.distill_losses = nn.ModuleDict()
        self.distill_cfg = distill_cfg

        self.with_logit=False
        self.local_buffer = {}
        """
            register hooks to cache the feature of FPN
        """
        student_modules = dict(self.student.named_modules())
        teacher_modules = dict(self.teacher.named_modules())

        def regitster_feature_hooks(
            student_module, teacher_module, output_hook=True, local_buffer=False
        ):
            def hook_teacher_forward(module, input, output):
                if output_hook:
                    self.register_buffer(teacher_module, output)
                else:
                    if local_buffer:
                        self.local_buffer[teacher_module].append(input)
                    else:
                        self.register_buffer(teacher_module, input)

            def hook_student_forward(module, input, output):
                if output_hook:
                    self.register_buffer(student_module, output)
                else:
                    if local_buffer:
                        self.local_buffer[student_module].append(input)
                    else:
                        self.register_buffer(student_module, input)

            return hook_teacher_forward, hook_student_forward

        for item_loc in distill_cfg:
            if item_loc.type=='logit':
                self.with_logit = True
            student_module = "student_" + item_loc.student_module.replace(".", "_")
            teacher_module = "teacher_" + item_loc.teacher_module.replace(".", "_")

            if item_loc.local_buffer:
                self.local_buffer[student_module] = []
                self.local_buffer[teacher_module] = []
            # else:
            #     self.register_buffer(student_module, None)
            #     self.register_buffer(teacher_module, None)

            hook_teacher_forward, hook_student_forward = regitster_feature_hooks(
                student_module,
                teacher_module,
                item_loc.output_hook,
                item_loc.local_buffer,
            )
            teacher_modules[item_loc.teacher_module].register_forward_hook(
                hook_teacher_forward
            )
            student_modules[item_loc.student_module].register_forward_hook(
                hook_student_forward
            )

            for item_loss in item_loc.methods:
                loss_name = item_loss.name
                if loss_name in self.distill_losses.keys():
                    continue
                self.distill_losses[loss_name] = build_distill_loss(item_loss)

    def base_parameters(self):
        return nn.ModuleList([self.student, self.distill_losses])

    @property
    def with_neck(self):
        """bool: whether the detector has a neck"""
        return hasattr(self.student, "neck") and self.student.neck is not None

    @property
    def with_shared_head(self):
        """bool: whether the detector has a shared head in the RoI Head"""
        return (
            hasattr(self.student, "roi_head") and self.student.roi_head.with_shared_head
        )

    @property
    def with_bbox(self):
        """bool: whether the detector has a bbox head"""
        return (
            hasattr(self.student, "roi_head") and self.student.roi_head.with_bbox
        ) or (hasattr(self.student, "bbox_head") and self.student.bbox_head is not None)

    @property
    def with_mask(self):
        """bool: whether the detector has a mask head"""
        return (
            hasattr(self.student, "roi_head") and self.student.roi_head.with_mask
        ) or (hasattr(self.student, "mask_head") and self.student.mask_head is not None)

    def init_weights_teacher(self, path=None):
        """Load the pretrained model in teacher detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        checkpoint = load_checkpoint(self.teacher, path, map_location="cpu")

    def forward_train(self, img, img_metas, **kwargs):

        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.

        Returns:
            dict[str, Tensor]: A dictionary of loss components(student's losses and distiller's losses).
        """
        
        with_adv = False
        if "adv" in kwargs.keys():
            with_adv = True
            adv_img = kwargs.pop("adv")
            img = torch.cat((img, adv_img), 0)
            img_metas = img_metas * 2
            for k, v in kwargs.items():
                v.extend(v)

        self.student.forward_train_step_1(img, img_metas)
        with torch.no_grad():
            self.teacher.eval()
            self.teacher.forward_train_step_1(img, img_metas)

        student_loss = {}
        clean_x_s = []
        adv_x_t = []
        adv_x_s = []
        # buffer_dict = dict(self.named_buffers())
        for item_loc in self.distill_cfg:
            if item_loc.type != 'feature':
                continue
            student_module = "student_" + item_loc.student_module.replace(".", "_")
            teacher_module = "teacher_" + item_loc.teacher_module.replace(".", "_")

            student_feat = self.get_buffer(student_module)
            teacher_feat = self.get_buffer(teacher_module)
            if with_adv:
                clean_feat_s, adv_feat_s = torch.chunk(student_feat, chunks=2, dim=0)
                clean_feat_t, adv_feat_t = torch.chunk(teacher_feat, chunks=2, dim=0)
                
                adv_x_s.append(adv_feat_s)
                adv_x_t.append(adv_feat_t)
            else:
                clean_feat_s = student_feat
                clean_feat_t  = teacher_feat
            clean_x_s.append(clean_feat_s)
            for item_loss in item_loc.methods:
                loss_name = item_loss.name
                if loss_name not in student_loss.keys():
                    student_loss[loss_name] = 0
                if str(loss_name).startswith("adv") and with_adv:
                    student_loss[loss_name] += self.distill_losses[loss_name](
                        adv_feat_s, adv_feat_t
                    )
                else:
                    student_loss[loss_name] += self.distill_losses[loss_name](
                        clean_feat_s, clean_feat_t, kwargs["gt_bboxes"], img_metas
                    )
        if with_adv:
            img_metas = img_metas[:len(img_metas)//2]
            for k, v in kwargs.items():
                kwargs[k] = v[:len(v)//2]
        student_loss.update(
            self.student.forward_train_step_2(clean_x_s, img_metas, **kwargs)
        )
        if not self.with_logit:
            return student_loss
        # clear logit buffer
        for k, v in self.local_buffer.items():
            self.local_buffer[k].clear()
        self.student.forward_train_step_2(adv_x_s, img_metas, **kwargs)
        with torch.no_grad():
            self.teacher.eval()
            self.teacher.forward_train_step_2(adv_x_t, img_metas, **kwargs)
        for item_loc in self.distill_cfg:
            if item_loc.type != 'logit':
                continue
            student_module = "student_" + item_loc.student_module.replace(".", "_")
            teacher_module = "teacher_" + item_loc.teacher_module.replace(".", "_")
            target = torch.cat([lgt[1] for lgt in self.local_buffer[student_module]])
            valid_idx = target!=len(self.CLASSES)
            target = target[valid_idx]
            student_logit = torch.cat([lgt[0] for lgt in self.local_buffer[student_module]])[valid_idx]
            teacher_logit = torch.cat([lgt[0] for lgt in self.local_buffer[teacher_module]])[valid_idx]
            
            for item_loss in item_loc.methods:
                loss_name  =  item_loss.name
                student_loss[loss_name] = self.distill_losses[loss_name](student_logit,teacher_logit,target)
            
        return student_loss

    def simple_test(self, img, img_metas, **kwargs):
        return self.student.simple_test(img, img_metas, **kwargs)

    def aug_test(self, imgs, img_metas, **kwargs):
        return self.student.aug_test(imgs, img_metas, **kwargs)

    def extract_feat(self, imgs):
        """Extract features from images."""
        return self.student.extract_feat(imgs)
