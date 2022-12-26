from .fgd_loss import  FGDLoss
from .cwd_loss import CWDLoss
from .mgd_loss import MGDLoss
from .adv_feature_loss import AdvFeatureLoss
from .fkd_loss import FKDLoss
from .logit_loss import DKDLoss,OriCELoss
__all__ = [
    'FGDLoss','CWDLoss' , 'MGDLoss','AdvFeatureLoss','FKDLoss','DKDLoss','OriCELoss'
]
