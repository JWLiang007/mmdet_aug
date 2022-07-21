import torch.nn as nn
import torch.nn.functional as F
import torch
from ..builder import DISTILL_LOSSES

@DISTILL_LOSSES.register_module()
class AdvFeatureLoss(nn.Module):

    """PyTorch version of `Masked Generative Distillation`
   
    Args:
        student_channels(int): Number of channels in the student's feature map.
        teacher_channels(int): Number of channels in the teacher's feature map. 
        name (str): the loss name of the layer
        alpha_mgd (float, optional): Weight of dis_loss. Defaults to 0.00002
        lambda_mgd (float, optional): masked ratio. Defaults to 0.65
    """
    def __init__(self,
                 student_channels,
                 teacher_channels,
                 name,
                 alpha_adv,
                 layer_idx
                 ):
        super(AdvFeatureLoss, self).__init__()
        self.name = name
        self.layer_idx = layer_idx
        self.alpha_adv = alpha_adv
        if student_channels != teacher_channels:
            self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.align = None


    def forward(self,
                feat_s,
                feat_t):
        """Forward function.
        Args:
            preds_S(Tensor): Bs*C*H*W, student's feature map
            preds_T(Tensor): Bs*C*H*W, teacher's feature map
        """
        preds_S = feat_s[self.layer_idx]
        preds_T = feat_t[self.layer_idx]
        assert preds_S.shape[-2:] == preds_T.shape[-2:]

        if self.align is not None:
            preds_S = self.align(preds_S)
    
        loss = self.get_dis_loss(preds_S, preds_T)*self.alpha_adv
            
        return loss

    def get_dis_loss(self, preds_S, preds_T):
        loss_mse = nn.MSELoss(reduction='sum')
        N, C, H, W = preds_T.shape

        dis_loss = loss_mse(preds_S, preds_T)/N

        return dis_loss