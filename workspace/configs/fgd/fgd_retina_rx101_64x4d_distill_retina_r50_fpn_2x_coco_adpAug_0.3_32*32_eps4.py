_base_ = [
    './fgd_retina_rx101_64x4d_distill_retina_r50_fpn_2x_coco.py'
]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile',adv_img='data/adv_coco_4_5'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='InstanceAug',prob=0.3),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

data = dict(

    train=dict(
        pipeline=train_pipeline),
)