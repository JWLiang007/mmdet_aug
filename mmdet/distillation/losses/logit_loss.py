import torch.nn as nn
import torch.nn.functional as F
import torch

from ..builder import DISTILL_LOSSES


@DISTILL_LOSSES.register_module()
class DKDLoss(nn.Module):

    def __init__(self,
                 name,
                alpha= 1.0, 
                beta=0.25,
                temp=1.0,
                
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

    def forward(self, raw_s , raw_t  ):
        
        logits_student, logits_teacher, target = self.preprocess_data(raw_s , raw_t )
        gt_mask = self._get_gt_mask(logits_student, target)
        other_mask = self._get_other_mask(logits_student, target)
        # softmax has beene done in preprocess_data
        # pred_student = F.softmax(logits_student / self.temp, dim=1)
        # pred_teacher = F.softmax(logits_teacher / self.temp, dim=1)
        pred_student = self.cat_mask(logits_student, gt_mask, other_mask)
        pred_teacher = self.cat_mask(logits_teacher, gt_mask, other_mask)
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
    
    def preprocess_data(self, raw_s , raw_t ):
        logit_stu = torch.cat([ r_i[0] for r_i in raw_s])
        # target_stu = torch.cat([ r_i[1] for r_i in raw_s])    # same as the target_tea
        weight_stu = torch.cat([ r_i[2] for r_i in raw_s])
        logit_tea = torch.cat([ r_i[0] for r_i in raw_t])
        target_gt = torch.cat([ r_i[1] for r_i in raw_t])
        weight_tea = torch.cat([ r_i[2] for r_i in raw_t])
        
        logit_stu = F.softmax(logit_stu / self.temp, dim=1)
        logit_tea = F.softmax(logit_tea / self.temp, dim=1)
        
        target_tea = logit_tea.max(1)
        mean_tea = target_tea[0].mean()
        std_tea = target_tea[0].std()
        tea_idct = target_tea[0]>mean_tea+std_tea
        gt_idct = target_gt!=logit_tea.shape[-1]
        target_gt[tea_idct] = target_tea[1][tea_idct]
        valid_idct = torch.logical_and(weight_stu,weight_tea)   # filter invalid proposals
        valid_idct = torch.logical_and(valid_idct , torch.logical_or(tea_idct,gt_idct) )  # filter bg proposals
        return logit_stu[valid_idct], logit_tea[valid_idct], target_gt[valid_idct]
        
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
    
@DISTILL_LOSSES.register_module()
class CELoss(nn.Module):
    
    def __init__(self,
                 alpha,
                 ):
        super(CELoss, self).__init__()
        self.alpha = alpha
    def forward(self,  logits_student, logits_teacher, target, ):
        norm_logits_teacher = F.softmax(logits_teacher , dim=1)
        ce_loss = F.cross_entropy(logits_student,norm_logits_teacher,) * self.alpha
        return ce_loss


@DISTILL_LOSSES.register_module()
class OriCELoss(nn.Module):
    
    def __init__(self,
                 ):
        super(OriCELoss, self).__init__()
    def forward(self, *loss_student ):
        return loss_student[0]