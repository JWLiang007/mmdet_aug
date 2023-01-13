_base_ = [
    './fgd_cascade_mask_rcnn_rx101_32x4d_distill_mask_rcnn_r50_fpn_2x_voc.py'
]

train_pipeline = [
    dict(type='LoadImageFromFile',adv_img='data/adv_voc_4_5'),
    dict(type='LoadAnnotations', with_bbox=True,with_mask=True),
    dict(type='InstanceAug',prob=0.7),
    dict(type='Resize', img_scale=(1000, 600), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='RandomCrop', crop_size=(0.8, 0.8), crop_type='relative_range', adaptive=True, bbox_size=(32, 32)),
    dict(type='Normalize',
         mean=[123.675, 116.28, 103.53],
         std=[58.395, 57.12, 57.375],
         to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]


data = dict(
    train=dict(
        pipeline=train_pipeline),
)