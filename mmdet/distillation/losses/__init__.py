from .fgd_loss import  FGDLoss
from .cwd_loss import CWDLoss
from .mgd_loss import MGDLoss
from .adv_feature_loss import AdvFeatureLoss,CtrFeatureLoss
from .fkd_loss import FKDLoss
from .logit_loss import DKDLoss,OriCELoss,CELoss
__all__ = [
    'FGDLoss','CWDLoss' , 'MGDLoss','AdvFeatureLoss','FKDLoss','DKDLoss','OriCELoss','CtrFeatureLoss','CELoss'
]
