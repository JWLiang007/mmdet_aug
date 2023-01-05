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
                #  name,
                 alpha_adv,
                #  layer_idx,
                 loss_type = 'mse',
                 **kwargs
                 ):
        super(AdvFeatureLoss, self).__init__()
        # self.name = name
        # self.layer_idx = layer_idx
        self.alpha_adv = alpha_adv
        self.loss_type = loss_type
        self.loss_param = kwargs
        if student_channels != teacher_channels:
            self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.align = None



    def forward(self,
                feat_s,
                feat_t,
                **kwargs):
        """Forward function.
        Args:
            preds_S(Tensor): Bs*C*H*W, student's feature map
            preds_T(Tensor): Bs*C*H*W, teacher's feature map
        """
        preds_S = feat_s
        preds_T = feat_t
        assert preds_S.shape[-2:] == preds_T.shape[-2:]

        if self.align is not None:
            preds_S = self.align(preds_S)
    
        loss = self.get_dis_loss(preds_S, preds_T,**kwargs)*self.alpha_adv
            
        return loss

    def get_dis_loss(self, preds_S, preds_T,**kwargs):

        N, C, H, W = preds_T.shape
        if self.loss_type == 'mse':
            loss_mse = nn.MSELoss(reduction='sum')
            loss = loss_mse(preds_S, preds_T)/N
        elif self.loss_type == 'l1':
            loss_mse = nn.L1Loss(reduction='sum')
            loss = loss_mse(preds_S, preds_T)/N
        elif self.loss_type == 'cwd':
            assert 'tau' in self.loss_param.keys()
            tau = self.loss_param['tau']
            softmax_pred_T = F.softmax(preds_T.view(-1, W * H) / tau, dim=1)

            logsoftmax = torch.nn.LogSoftmax(dim=1)
            loss = torch.sum(softmax_pred_T *
                             logsoftmax(preds_T.view(-1, W * H) / tau) -
                             softmax_pred_T *
                             logsoftmax(preds_S.view(-1, W * H) / tau)) * (
                                 tau**2)

            loss =  loss / (C * N)
        return loss


@DISTILL_LOSSES.register_module()
class CtrFeatureLoss(nn.Module):

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
                #  name,
                 alpha_ctr,
                #  layer_idx,
                 with_discp = True,
                 loss_type = 'mse',
                 **kwargs
                 ):
        super(CtrFeatureLoss, self).__init__()
        # self.name = name
        # self.layer_idx = layer_idx
        self.alpha_ctr = alpha_ctr
        self.with_discp = with_discp
        self.loss_type = loss_type
        self.loss_param = kwargs
        if student_channels != teacher_channels:
            self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.align = None



    def forward(self,
                feat_s,
                feat_t,
                **kwargs):
        """Forward function.
        Args:
            preds_S(Tensor): Bs*C*H*W, student's feature map
            preds_T(Tensor): Bs*C*H*W, teacher's feature map
        """
        
        clean_s = feat_s['clean'][0].clone().detach()
        clean_t = feat_t['clean'][0]
        adv_s = feat_s['adv'][0]
        adv_t = feat_t['adv'][0]

        assert clean_s.shape[-2:] == clean_t.shape[-2:] and adv_s.shape[-2:] == adv_t.shape[-2:]

        if self.align is not None:
            clean_s = self.align(clean_s)
            adv_s = self.align(adv_s)
        n = clean_s.shape[0]
        if self.loss_type == 'contrastive':
            
            
            clean_s = F.softmax(clean_s.view([n,-1]),dim=1)  
            adv_s = F.softmax(adv_s.view([n,-1]),dim=1)  
            adv_t = F.softmax(adv_t.view([n,-1]),dim=1)  
            similarity = torch.exp(F.cosine_similarity(adv_s, adv_t, dim=1))
            if self.with_discp:
                discrepancy = torch.exp(F.cosine_similarity(adv_s, clean_s, dim=1))
            else:
                discrepancy = 1.
            loss = torch.sum(-torch.log(similarity/discrepancy)) *self.alpha_ctr/ n 
            # loss = self.get_dis_loss(adv_s, adv_t,**kwargs)*self.alpha_ctr / self.get_dis_loss(adv_s, clean_s,**kwargs)
        elif self.loss_type == 'mse':
            
            loss = self.get_dis_loss(adv_s, adv_t,**kwargs) * self.alpha_ctr
        return loss
    
    def get_dis_loss(self, preds_S, preds_T,**kwargs):

        N, C, H, W = preds_T.shape
        if self.loss_type == 'mse':
            loss_mse = nn.MSELoss(reduction='sum')
            loss = loss_mse(preds_S, preds_T)/N
        elif self.loss_type == 'l1':
            loss_mse = nn.L1Loss(reduction='sum')
            loss = loss_mse(preds_S, preds_T)/N
        elif self.loss_type == 'cwd':
            assert 'tau' in self.loss_param.keys()
            tau = self.loss_param['tau']
            softmax_pred_T = F.softmax(preds_T.view(-1, W * H) / tau, dim=1)

            logsoftmax = torch.nn.LogSoftmax(dim=1)
            loss = torch.sum(softmax_pred_T *
                             logsoftmax(preds_T.view(-1, W * H) / tau) -
                             softmax_pred_T *
                             logsoftmax(preds_S.view(-1, W * H) / tau)) * (
                                 tau**2)

            loss =  loss / (C * N)
        return loss