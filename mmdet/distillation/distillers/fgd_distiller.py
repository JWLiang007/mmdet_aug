import torch.nn as nn
import torch
from mmdet.models.detectors.base import BaseDetector
from mmdet.models.detectors.two_stage import TwoStageDetector
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint, _load_checkpoint, load_state_dict
from ..builder import DISTILLER, build_distill_loss
from collections import OrderedDict
import torch.nn.functional as F


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
            t_checkpoint = _load_checkpoint(
                teacher_pretrained, map_location="cpu")
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

        self.img_type = "clean"
        self.with_clean_logit = False
        self.with_adv_logit = False
        self.with_clean_feature = False
        self.with_adv_feature = False
        self.local_buffer = {}
        """
            register hooks to cache the feature of FPN
        """
        student_modules = dict(self.student.named_modules())
        teacher_modules = dict(self.teacher.named_modules())

        def regitster_hooks(
            student_module,
            teacher_module,
            hook_type="output",
            img_type="clean",
        ):
            img_type_list = img_type if isinstance(img_type,
                                                   list) else [img_type]

            def hook_teacher_forward(module, input, output):
                if self.img_type in img_type_list and self.run_mode == 'train':
                    module_key = teacher_module
                    # + "_" + '_'.join(img_type_list)
                    if module_key not in self.local_buffer.keys():
                        self.local_buffer[module_key] = {}
                    if self.img_type not in self.local_buffer[module_key].keys(
                    ):
                        self.local_buffer[module_key][self.img_type] = []
                    if hook_type == "input":
                        self.local_buffer[module_key][self.img_type].append(
                            input)
                    elif hook_type == "output":
                        self.local_buffer[module_key][self.img_type].append(
                            output)

            def hook_student_forward(module, input, output):
                if self.img_type in img_type_list and self.run_mode == 'train':
                    module_key = student_module
                    # + "_" + '_'.join(img_type_list)
                    if module_key not in self.local_buffer.keys():
                        self.local_buffer[module_key] = {}
                    if self.img_type not in self.local_buffer[module_key].keys(
                    ):
                        self.local_buffer[module_key][self.img_type] = []
                    if hook_type == "input":
                        self.local_buffer[module_key][self.img_type].append(
                            input)
                    elif hook_type == "output":
                        self.local_buffer[module_key][self.img_type].append(
                            output)

            return hook_teacher_forward, hook_student_forward

        for item_loc in distill_cfg:

            student_module = "student_" + item_loc.student_module.replace(
                ".", "_")
            teacher_module = "teacher_" + item_loc.teacher_module.replace(
                ".", "_")

            for item_loss in item_loc.methods:
                loss_name = item_loss.name
                if loss_name in self.distill_losses.keys():
                    continue
                loss_param = item_loss.loss_param
                self.distill_losses[loss_name] = build_distill_loss(loss_param)
                self.set_loss_flag(item_loss=item_loss)
                hook_teacher_forward, hook_student_forward = regitster_hooks(
                    student_module,
                    teacher_module,
                    item_loss.hook_type,
                    item_loss.img_type,
                )
                teacher_modules[item_loc.teacher_module].register_forward_hook(
                    hook_teacher_forward)
                student_modules[item_loc.student_module].register_forward_hook(
                    hook_student_forward)

    def set_loss_flag(self, item_loss):
        loss_input_type = item_loss.loss_input_type
        img_type = item_loss.img_type
        img_type_list = img_type if isinstance(img_type, list) else [img_type]
        assert loss_input_type in ["logit", "feature"]
        assert set(img_type_list).issubset(["clean", "adv"])
        if loss_input_type == "logit":
            if "clean" in img_type_list:
                self.with_clean_logit = True
            if "adv" in img_type_list:
                self.with_adv_logit = True
        elif loss_input_type == "feature":
            if "clean" in img_type_list:
                self.with_clean_feature = True
            if "adv" in img_type_list:
                self.with_adv_feature = True

    def preprocess_loss_input(
        self,
        loss_input_s,
        loss_input_t,
        item_loss,
        img_metas,
        **kwargs,
    ):
        img_type = item_loss.img_type
        img_type_list = img_type if isinstance(img_type, list) else [img_type]
        loss_input_type = item_loss.loss_input_type

        if len(img_type_list) == 1:
            if 'adv' in img_type_list:
                loss_input_s = loss_input_s['adv']
                loss_input_t = loss_input_t['adv']
            elif 'clean' in img_type_list:
                loss_input_s = loss_input_s['clean']
                loss_input_t = loss_input_t['clean']

        if loss_input_type == 'feature':
            if item_loss.loss_param.type == "AdvFeatureLoss":
                return [loss_input_s[0], loss_input_t[0]]
            if item_loss.loss_param.type == "FGDLoss":
                return [
                    loss_input_s[0], loss_input_t[0], kwargs["gt_bboxes"],
                    img_metas
                ]
            if item_loss.loss_param.type == "CtrFeatureLoss":
                return [loss_input_s, loss_input_t]
        elif loss_input_type == 'logit':
            if item_loss.loss_param.type == 'OriCELoss':
                return [loss_input_s]
            logit_filter = item_loss.logit_filter
            assert logit_filter in ['teacher', 'gt']
            teacher_logit = torch.cat([lgt[0] for lgt in loss_input_t])
            student_logit = torch.cat([lgt[0] for lgt in loss_input_s])
            target = torch.cat([lgt[1] for lgt in loss_input_s])
            weights = torch.cat([lgt[2] for lgt in loss_input_s])
            valid_ind = weights != 0
            if logit_filter == "gt":
                valid_ind = torch.logical_and(valid_ind,
                                              target != len(self.CLASSES))
                target = target[valid_ind]
            elif logit_filter == "teacher":
                _teacher_logit = F.softmax(teacher_logit, 1)
                val, idx = torch.max(_teacher_logit, 1)
                p_n_ind = torch.zeros_like(val).bool()
                # set positive anchor
                threshold_p = item_loss.get('threshold_p',0.7)
                pos_idx = torch.where(val >= threshold_p)[0]
                p_n_ind[pos_idx] = True
                # set negative anchor
                if item_loss.get('with_neg', False):
                    threshold_n = item_loss.get('threshold_n',0.3)
                    neg_idx = torch.where(val <= threshold_n)[0]
                    neg_idx = neg_idx[torch.randint(neg_idx.shape[0],
                                                    (pos_idx.shape[0] * 3, ))]
                    p_n_ind[neg_idx] = True
                if item_loss.get('with_gt', False):
                    p_n_ind = torch.logical_or(p_n_ind,
                                               target != len(self.CLASSES))
                valid_ind = torch.logical_and(valid_ind, p_n_ind)
                target = idx[valid_ind]
            student_logit = student_logit[valid_ind]
            teacher_logit = teacher_logit[valid_ind]
            return [student_logit, teacher_logit, target]

    def base_parameters(self):
        return nn.ModuleList([self.student, self.distill_losses])

    @property
    def with_neck(self):
        """bool: whether the detector has a neck"""
        return hasattr(self.student, "neck") and self.student.neck is not None

    @property
    def with_shared_head(self):
        """bool: whether the detector has a shared head in the RoI Head"""
        return (hasattr(self.student, "roi_head")
                and self.student.roi_head.with_shared_head)

    @property
    def with_bbox(self):
        """bool: whether the detector has a bbox head"""
        return (hasattr(self.student, "roi_head")
                and self.student.roi_head.with_bbox) or (
                    hasattr(self.student, "bbox_head")
                    and self.student.bbox_head is not None)

    @property
    def with_mask(self):
        """bool: whether the detector has a mask head"""
        return (hasattr(self.student, "roi_head")
                and self.student.roi_head.with_mask) or (
                    hasattr(self.student, "mask_head")
                    and self.student.mask_head is not None)

    def init_weights_teacher(self, path=None):
        """Load the pretrained model in teacher detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        checkpoint = load_checkpoint(self.teacher, path, map_location="cpu")

    def empty_buffer(self):
        for k, v in self.local_buffer.items():
            for i in reversed(range(len(v))):
                del v[i]

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

        # for k, v in self.local_buffer.items():
        #     self.local_buffer[k] = []
        # self.student.zero_grad()
        # with_adv = False
        if "adv" in kwargs.keys():
            # with_adv = True
            adv_img = kwargs.pop("adv")
            # img = torch.cat((img, adv_img), 0)
            # img_metas = img_metas * 2
            # for k, v in kwargs.items():
            #     v.extend(v)

        self.img_type = "clean"
        student_loss = self.student.forward_train(img, img_metas, **kwargs)
        if self.with_clean_logit:
            with torch.no_grad():
                self.teacher.forward_train(img, img_metas, **kwargs)
        elif self.with_clean_feature:
            with torch.no_grad():
                self.teacher.forward_train_step_1(img, img_metas)
        self.img_type = 'adv'
        if self.with_adv_logit:
            kwargs_s = kwargs.copy()
            kwargs_t = kwargs.copy()
            with torch.no_grad():
                if  isinstance(self.teacher,TwoStageDetector):
                    kwargs_t['return_proposals'] = True
                    proposals = self.teacher.forward_train(adv_img, img_metas, **kwargs_t)
            if isinstance(self.student,TwoStageDetector):
                kwargs_t['override_proposals'] = True
                kwargs_t['proposals'] = proposals
                self.student.forward_train(adv_img, img_metas, **kwargs_s)

        elif self.with_adv_feature:
            self.student.forward_train_step_1(adv_img, img_metas)
            with torch.no_grad():
                self.teacher.forward_train_step_1(adv_img, img_metas)
        for item_loc in self.distill_cfg:
            for item_loss in item_loc.methods:

                img_type = item_loss.img_type
                img_type_list = img_type if isinstance(img_type,
                                                       list) else [img_type]
                assert set(img_type_list).issubset(["clean", "adv"])
                loss_name = item_loss.name
                student_module = "student_" + item_loc.student_module.replace(
                    ".", "_")
                #   + "_" + '_'.join(img_type_list)
                teacher_module = "teacher_" + item_loc.teacher_module.replace(
                    ".", "_")
                #   + "_" + '_'.join(img_type_list)
                loss_name = item_loss.name
                assert loss_name not in student_loss.keys()
                # student_loss[loss_name] = torch.zeros(1).cuda()
                assert (len(self.local_buffer[teacher_module]) != 0
                        and len(self.local_buffer[student_module]) != 0)
                loss_input_t = self.local_buffer[teacher_module]
                loss_input_s = self.local_buffer[student_module]
                loss_input = self.preprocess_loss_input(
                    loss_input_s,
                    loss_input_t,
                    item_loss,
                    img_metas,
                    **kwargs,
                )
                assert loss_name not in student_loss
                student_loss[loss_name] = self.distill_losses[loss_name](
                    *loss_input)
        for k, v in self.local_buffer.items():
            for k1, v1 in v.items():
                v[k1] = []
            # self.local_buffer[k] = []
        # self.empty_buffer()
        return student_loss
        # with torch.no_grad():
        #     self.teacher.eval()
        #     tmp_feat_t = self.teacher.forward_train_step_1(img, img_metas)

        # student_loss = {}
        # clean_x_s = []
        # clean_x_t = []
        # adv_x_t = []
        # adv_x_s = []
        # # buffer_dict = dict(self.named_buffers())
        # for item_loc in self.distill_cfg:
        #     if item_loc.type != "feature":
        #         continue
        #     postfix = "" if not hasattr(item_loc,
        #                                 "input_type") else item_loc.input_type
        #     student_module = ("student_" +
        #                       item_loc.student_module.replace(".", "_") + "_" +
        #                       postfix)
        #     teacher_module = ("teacher_" +
        #                       item_loc.teacher_module.replace(".", "_") + "_" +
        #                       postfix)

        #     student_feat = self.get_buffer(student_module)
        #     teacher_feat = self.get_buffer(teacher_module)
        #     self.register_buffer(teacher_module, None)
        #     self.register_buffer(student_module, None)
        #     if with_adv:
        #         clean_feat_s, adv_feat_s = torch.chunk(
        #             student_feat, chunks=2, dim=0)
        #         clean_feat_t, adv_feat_t = torch.chunk(
        #             teacher_feat, chunks=2, dim=0)

        #         adv_x_s.append(adv_feat_s)
        #         adv_x_t.append(adv_feat_t)
        #     else:
        #         clean_feat_s = student_feat
        #         clean_feat_t = teacher_feat
        #     clean_x_s.append(clean_feat_s)
        #     clean_x_t.append(clean_feat_t)
        #     for item_loss in item_loc.methods:
        #         loss_name = item_loss.name
        #         if loss_name not in student_loss.keys():
        #             student_loss[loss_name] = torch.zeros(1).cuda()
        #         if str(loss_name).startswith("adv") and with_adv:
        #             student_loss[loss_name] += self.distill_losses[loss_name](
        #                 adv_feat_s, adv_feat_t)
        #         else:
        #             student_loss[loss_name] += self.distill_losses[loss_name](
        #                 clean_feat_s, clean_feat_t, kwargs["gt_bboxes"],
        #                 img_metas)
        # if with_adv:
        #     img_metas = img_metas[:len(img_metas) // 2]
        #     for k, v in kwargs.items():
        #         kwargs[k] = v[:len(v) // 2]
        # self.input_type = "clean"
        # student_loss.update(
        #     self.student.forward_train_step_2(clean_x_s, img_metas, **kwargs))
        # if not self.with_logit:
        #     return student_loss
        # clear logit buffer
        # for k, v in self.local_buffer.items():
        #     self.local_buffer[k] = []
        # self.student.forward_train_step_2(adv_x_s, img_metas, **kwargs)
        # with torch.no_grad():
        #     self.teacher.eval()
        #     self.teacher.forward_train_step_2(adv_x_t, img_metas, **kwargs)
        # for item_loc in self.distill_cfg:
        #     if item_loc.type != "logit":
        #         continue
        #     postfix = "" if not hasattr(item_loc,
        #                                 "input_type") else item_loc.input_type
        #     if postfix == "adv":
        #         self.input_type = "adv"
        #         self.student.forward_train_step_2(adv_x_s, img_metas, **kwargs)
        #         with torch.no_grad():
        #             self.teacher.eval()
        #             self.teacher.forward_train_step_2(adv_x_t, img_metas,
        #                                               **kwargs)
        #     elif postfix == "clean":
        #         self.input_type = "clean"
        #         with torch.no_grad():
        #             self.teacher.eval()
        #             self.teacher.forward_train_step_2(clean_x_t, img_metas,
        #                                               **kwargs)
        #     student_module = ("student_" +
        #                       item_loc.student_module.replace(".", "_") + "_" +
        #                       postfix)
        #     teacher_module = ("teacher_" +
        #                       item_loc.teacher_module.replace(".", "_") + "_" +
        #                       postfix)
        #     teacher_logit = torch.cat(
        #         [lgt[0] for lgt in self.local_buffer[teacher_module]])
        #     student_logit = torch.cat(
        #         [lgt[0] for lgt in self.local_buffer[student_module]])
        #     target = torch.cat(
        #         [lgt[1] for lgt in self.local_buffer[student_module]])
        #     weights = torch.cat(
        #         [lgt[2] for lgt in self.local_buffer[student_module]])
        #     valid_idx = weights != 0
        #     assert hasattr(item_loc, "logit_filter")
        #     if item_loc.logit_filter == "gt":
        #         valid_idx = torch.logical_and(valid_idx,
        #                                       target != len(self.CLASSES))
        #         target = target[valid_idx]
        #     elif item_loc.logit_filter == "teacher":
        #         _teacher_logit = F.softmax(teacher_logit, 1)
        #         val, idx = torch.max(_teacher_logit, 1)
        #         valid_idx = torch.logical_and(valid_idx, val > 0.7)
        #         target = idx[valid_idx]
        #     student_logit = student_logit[valid_idx]
        #     teacher_logit = teacher_logit[valid_idx]

        #     for item_loss in item_loc.methods:
        #         loss_name = item_loss.name
        #         student_loss[loss_name] = self.distill_losses[loss_name](
        #             student_logit, teacher_logit, target)
        # for k, v in self.local_buffer.items():
        #     self.local_buffer[k] = []
        # return student_loss

    def simple_test(self, img, img_metas, **kwargs):
        return self.student.simple_test(img, img_metas, **kwargs)

    def aug_test(self, imgs, img_metas, **kwargs):
        return self.student.aug_test(imgs, img_metas, **kwargs)

    def extract_feat(self, imgs):
        """Extract features from images."""
        return self.student.extract_feat(imgs)
