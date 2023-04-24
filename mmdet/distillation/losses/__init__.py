from .fgd_loss import  FGDLoss

from .mgd_loss import MGDLoss
from .adv_feature_loss import AdvFeatureLoss
from .fkd_loss import FKDLoss
from .cwd_loss import ChannelWiseDivergence
from .logit_loss import DKDLoss
__all__ = [
    'FGDLoss' , 'MGDLoss','AdvFeatureLoss','FKDLoss', 'ChannelWiseDivergence', 'DKDLoss'
]
