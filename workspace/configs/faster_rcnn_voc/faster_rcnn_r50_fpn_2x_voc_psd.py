_base_ = './faster_rcnn_r50_fpn_2x_voc.py'


# ======================

data_root = 'data/VOCdevkit/'
data = dict(

    train=dict(
        ann_file=data_root +'voc07_trainval_cascade_rx101.json',)
)