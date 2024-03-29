import torch.nn as nn
import torch.nn.functional as F
import torch
from mmcv.cnn import constant_init, kaiming_init
from ..builder import DISTILL_LOSSES

@DISTILL_LOSSES.register_module()
class FKDLoss(nn.Module):

    """PyTorch version of `Focal and Global Knowledge Distillation for Detectors`
   
    Args:
        student_channels(int): Number of channels in the student's feature map.
        teacher_channels(int): Number of channels in the teacher's feature map. 
        temp (float, optional): Temperature coefficient. Defaults to 0.5.
        name (str): the loss name of the layer
        alpha_fgd (float, optional): Weight of fg_loss. Defaults to 0.001
        beta_fgd (float, optional): Weight of bg_loss. Defaults to 0.0005
        gamma_fgd (float, optional): Weight of mask_loss. Defaults to 0.001
        lambda_fgd (float, optional): Weight of relation_loss. Defaults to 0.000005
    """
    def __init__(self,
                 # student_channels,
                 # teacher_channels,
                 name,
                 alpha_fkd=4e-4,
                 beta_fkd=2e-2,
                 gamma_fkd=4e-4,
                 T_fkd=0.5,
                 ):
        super(FKDLoss, self).__init__()

        # if student_channels != teacher_channels:
        #     self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)
        # else:
        #     self.align = None
        self.alpha_fkd = alpha_fkd
        self.beta_fkd = beta_fkd
        self.gamma_fkd = gamma_fkd
        self.T_fkd = T_fkd

        self.channel_wise_adaptation = nn.ModuleList([
            nn.Linear(256, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 256)
        ])

        self.spatial_wise_adaptation = nn.ModuleList([
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
        ])
        self.adaptation_layers = nn.ModuleList([
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
        ])
        self.student_non_local = nn.ModuleList(
            [
                NonLocalBlockND(in_channels=256, inter_channels=64, downsample_stride=8),
                NonLocalBlockND(in_channels=256, inter_channels=64, downsample_stride=4),
                NonLocalBlockND(in_channels=256),
                NonLocalBlockND(in_channels=256),
                NonLocalBlockND(in_channels=256)
            ]
        )
        self.teacher_non_local = nn.ModuleList(
            [
                NonLocalBlockND(in_channels=256, inter_channels=64, downsample_stride=8),
                NonLocalBlockND(in_channels=256, inter_channels=64, downsample_stride=4),
                NonLocalBlockND(in_channels=256),
                NonLocalBlockND(in_channels=256),
                NonLocalBlockND(in_channels=256)
            ]
        )
        self.non_local_adaptation = nn.ModuleList([
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        ])


    def forward(self,
                preds_S,
                preds_T,
                layer_idx,
                is_two_stage = False,
                ):
        """Forward function.
        Args:
            preds_S(Tensor): Bs*C*H*W, student's feature map
            preds_T(Tensor): Bs*C*H*W, teacher's feature map
            gt_bboxes(tuple): Bs*[nt*4], pixel decimal: (tl_x, tl_y, br_x, br_y)
            img_metas (list[dict]): Meta information of each image, e.g.,
            image size, scaling factor, etc.
        """
        assert preds_S.shape[-2:] == preds_T.shape[-2:],'the output dim of teacher and student differ'

        # if self.align is not None:
        #     preds_S = self.align(preds_S)
        #
        # N,C,H,W = preds_S.shape

        t = self.T_fkd
        s_ratio = 1.0
        c_t = self.T_fkd
        c_s_ratio = 1.0

        x = preds_S
        losses = list()
        # losses = dict()
        kd_feat_loss = 0
        kd_channel_loss = 0
        kd_spatial_loss = 0


        t_feats = preds_T
        # for _i in range(len(t_feats)):
        t_attention_mask = torch.mean(torch.abs(t_feats), [1], keepdim=True)
        size = t_attention_mask.size()
        t_attention_mask = t_attention_mask.view(x.size(0), -1)
        t_attention_mask = torch.softmax(t_attention_mask / t, dim=1) * size[-1] * size[-2]
        t_attention_mask = t_attention_mask.view(size)

        s_attention_mask = torch.mean(torch.abs(x), [1], keepdim=True)
        size = s_attention_mask.size()
        s_attention_mask = s_attention_mask.view(x.size(0), -1)
        s_attention_mask = torch.softmax(s_attention_mask / t, dim=1) * size[-1] * size[-2]
        s_attention_mask = s_attention_mask.view(size)

        c_t_attention_mask = torch.mean(torch.abs(t_feats), [2, 3], keepdim=True)  # 2 x 256 x 1 x1
        c_size = c_t_attention_mask.size()
        c_t_attention_mask = c_t_attention_mask.view(x.size(0), -1)  # 2 x 256
        c_t_attention_mask = torch.softmax(c_t_attention_mask / c_t, dim=1) * 256
        c_t_attention_mask = c_t_attention_mask.view(c_size)  # 2 x 256 -> 2 x 256 x 1 x 1

        c_s_attention_mask = torch.mean(torch.abs(x), [2, 3], keepdim=True)  # 2 x 256 x 1 x1
        c_size = c_s_attention_mask.size()
        c_s_attention_mask = c_s_attention_mask.view(x.size(0), -1)  # 2 x 256
        c_s_attention_mask = torch.softmax(c_s_attention_mask / c_t, dim=1) * 256
        c_s_attention_mask = c_s_attention_mask.view(c_size)  # 2 x 256 -> 2 x 256 x 1 x 1

        sum_attention_mask = (t_attention_mask + s_attention_mask * s_ratio) / (1 + s_ratio)
        sum_attention_mask = sum_attention_mask.detach()

        c_sum_attention_mask = (c_t_attention_mask + c_s_attention_mask * c_s_ratio) / (1 + c_s_ratio)
        c_sum_attention_mask = c_sum_attention_mask.detach()
        if is_two_stage :
            c_sum_attention_mask = None
        kd_feat_loss += self.dist2(t_feats, self.adaptation_layers[layer_idx](x), attention_mask=sum_attention_mask,
                              channel_attention_mask=c_sum_attention_mask) * self.beta_fkd
        kd_channel_loss += torch.dist(torch.mean(t_feats, [2, 3]),
                                      self.channel_wise_adaptation[layer_idx](torch.mean(x, [2, 3]))) * self.alpha_fkd
        t_spatial_pool = torch.mean(t_feats, [1]).view(t_feats.size(0), 1, t_feats.size(2),
                                                           t_feats.size(3))
        s_spatial_pool = torch.mean(x, [1]).view(x.size(0), 1, x.size(2),
                                                     x.size(3))
        kd_spatial_loss += torch.dist(t_spatial_pool, self.spatial_wise_adaptation[layer_idx](s_spatial_pool)) * self.alpha_fkd

        losses.append(kd_feat_loss)
        losses.append(kd_channel_loss)
        losses.append(kd_spatial_loss)

        kd_nonlocal_loss = 0

        t_feats  = preds_T
        # for _i in range(len(t_feats)):
        s_relation = self.student_non_local[layer_idx](x)
        t_relation = self.teacher_non_local[layer_idx](t_feats)
        #   print(s_relation.size())
        kd_nonlocal_loss += torch.dist(self.non_local_adaptation[layer_idx](s_relation), t_relation, p=2)
        losses.append(kd_nonlocal_loss * self.gamma_fkd)

        return losses

    def dist2(self,tensor_a, tensor_b, attention_mask=None, channel_attention_mask=None):
        diff = (tensor_a - tensor_b) ** 2
        #   print(diff.size())      batchsize x 1 x W x H,
        #   print(attention_mask.size()) batchsize x 1 x W x H
        diff = diff * attention_mask
        if channel_attention_mask is not None:
            diff = diff * channel_attention_mask
        diff = torch.sum(diff) ** 0.5
        return diff

class NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=2, sub_sample=True, bn_layer=True,
                 downsample_stride=2):
        super(NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(downsample_stride, downsample_stride))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :
        :
        '''

        batch_size = x.size(0)  # 2 , 256 , 300 , 300

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)  # 2 , 128 , 150 x 150
        g_x = g_x.permute(0, 2, 1)  # 2 , 150 x 150, 128

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)  # 2 , 128 , 300 x 300
        theta_x = theta_x.permute(0, 2, 1)  # 2 , 300 x 300 , 128
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)  # 2 , 128 , 150 x 150
        f = torch.matmul(theta_x, phi_x)  # 2 , 300x300 , 150x150
        N = f.size(-1)  # 150 x 150
        f_div_C = f / N  # 2 , 300x300, 150x150

        y = torch.matmul(f_div_C, g_x)  # 2, 300x300, 128
        y = y.permute(0, 2, 1).contiguous()  # 2, 128, 300x300
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z

