_base_ = './faster_rcnn_r50_fpn_2x_voc.py'


# ======================

data_root = 'data/VOCdevkit/'
data = dict(

    train=dict(
        ann_file=data_root +'cascade_rx101_voc07_trainval_0_7.json',)
)