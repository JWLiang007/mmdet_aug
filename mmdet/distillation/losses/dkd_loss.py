import torch.nn as nn
import torch.nn.functional as F
import torch

from ..builder import DISTILL_LOSSES


@DISTILL_LOSSES.register_module()
class DKDLoss(nn.Module):

    def __init__(self,
                #  name,
                alpha= 1.0, 
                beta=0.25,
                temp=1.0
                 ):
        super(DKDLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.temp = temp
        # self.tau = tau
        # self.loss_weight = weight
    
        # if student_channels != teacher_channels:
        #     self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)
        # else:
        #     self.align = None

    def forward(self, logits_student, logits_teacher, target, ):
        gt_mask = self._get_gt_mask(logits_student, target)
        other_mask = self._get_other_mask(logits_student, target)
        pred_student = F.softmax(logits_student / self.temp, dim=1)
        pred_teacher = F.softmax(logits_teacher / self.temp, dim=1)
        pred_student = self.cat_mask(pred_student, gt_mask, other_mask)
        pred_teacher = self.cat_mask(pred_teacher, gt_mask, other_mask)
        log_pred_student = torch.log(pred_student)
        tckd_loss = (
            F.kl_div(log_pred_student, pred_teacher, size_average=False)
            * (self.temp**2)
            / target.shape[0]
        )
        pred_teacher_part2 = F.softmax(
            logits_teacher / self.temp - 1000.0 * gt_mask, dim=1
        )
        log_pred_student_part2 = F.log_softmax(
            logits_student / self.temp - 1000.0 * gt_mask, dim=1
        )
        nckd_loss = (
            F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
            * (self.temp**2)
            / target.shape[0]
        )
        return self.alpha * tckd_loss + self.beta * nckd_loss


    def _get_gt_mask(self,logits, target):
        target = target.reshape(-1)
        mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
        return mask


    def _get_other_mask(self,logits, target):
        target = target.reshape(-1)
        mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
        return mask


    def cat_mask(self, t, mask1, mask2):
        t1 = (t * mask1).sum(dim=1, keepdims=True)
        t2 = (t * mask2).sum(1, keepdims=True)
        rt = torch.cat([t1, t2], dim=1)
        return rt
    
    # def forward(self,
    #             preds_S,
    #             preds_T):
    #     """Forward function."""
    #     assert preds_S.shape[-2:] == preds_T.shape[-2:],'the output dim of teacher and student differ'
    #     N,C,W,H = preds_S.shape

    #     if self.align is not None:
    #         preds_S = self.align(preds_S)

    #     softmax_pred_T = F.softmax(preds_T.view(-1, W * H) / self.tau, dim=1)

    #     logsoftmax = torch.nn.LogSoftmax(dim=1)
    #     loss = torch.sum(softmax_pred_T *
    #                      logsoftmax(preds_T.view(-1, W * H) / self.tau) -
    #                      softmax_pred_T *
    #                      logsoftmax(preds_S.view(-1, W * H) / self.tau)) * (
    #                          self.tau**2)

    #     loss = self.loss_weight * loss / (C * N)

    #     return loss



