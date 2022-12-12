_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

alpha_adv = 0.000000125
loss_type = 'mse'

teacher_ckpt = 'https://download.openmmlab.com/mmdetection/v2.0/gfl/gfl_r101_fpn_mstrain_2x_coco/gfl_r101_fpn_mstrain_2x_coco_20200629_200126-dd12f847.pth'  # noqa
model = dict(
    type='KnowledgeDistillationSingleStageDetector',
    teacher_config='configs/gfl/gfl_r101_fpn_mstrain_2x_coco.py',
    teacher_ckpt=teacher_ckpt,
    distill_cfg=[
        dict(student_module='neck.fpn_convs.4.conv',
             teacher_module='neck.fpn_convs.4.conv',
             output_hook=True,
             methods=[dict(type='AdvFeatureLoss',
                           name='adv_loss_mgd_fpn_4',
                           student_channels=256,
                           teacher_channels=256,
                           alpha_adv=alpha_adv,
                           layer_idx=4,
                           loss_type=loss_type,
                           )
                      ]
             ),
        dict(student_module='neck.fpn_convs.3.conv',
             teacher_module='neck.fpn_convs.3.conv',
             output_hook=True,
             methods=[dict(type='AdvFeatureLoss',
                           name='adv_loss_mgd_fpn_3',
                           student_channels=256,
                           teacher_channels=256,
                           alpha_adv=alpha_adv,
                           layer_idx=3,
                           loss_type=loss_type,
                           )
                      ]
             ),
        dict(student_module='neck.fpn_convs.2.conv',
             teacher_module='neck.fpn_convs.2.conv',
             output_hook=True,
             methods=[dict(type='AdvFeatureLoss',
                           name='adv_loss_mgd_fpn_2',
                           student_channels=256,
                           teacher_channels=256,
                           alpha_adv=alpha_adv,
                           layer_idx=2,
                           loss_type=loss_type,
                           )]
             ),
        dict(student_module='neck.fpn_convs.1.conv',
             teacher_module='neck.fpn_convs.1.conv',
             output_hook=True,
             methods=[dict(type='AdvFeatureLoss',
                           name='adv_loss_mgd_fpn_1',
                           student_channels=256,
                           teacher_channels=256,
                           alpha_adv=alpha_adv,
                           layer_idx=1,
                           loss_type=loss_type,
                           )
                      ]
             ),
        dict(student_module='neck.fpn_convs.0.conv',
             teacher_module='neck.fpn_convs.0.conv',
             output_hook=True,
             methods=[
                 dict(type='AdvFeatureLoss',
                      name='adv_loss_mgd_fpn_0',
                      student_channels=256,
                      teacher_channels=256,
                      alpha_adv=alpha_adv,
                      layer_idx=0,
                      loss_type=loss_type,
                      )
             ]),
    ],
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 256, 512],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='LDHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
        loss_ld=dict(
            type='KnowledgeDistillationKLDivLoss', loss_weight=0.25, T=10),
        reg_max=16,
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', adv_img='data/adv_coco_8_5/'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='InstanceAug', prob=0.3, subst_full=True, subst_stg='2'),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='RandomCrop', crop_size=(0.8, 0.8), crop_type='relative_range', adaptive=True, bbox_size=(32, 32),
         subst_stg='2'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'adv', 'gt_bboxes', 'gt_labels']),
]

data = dict(
    train=dict(
        pipeline=train_pipeline),
)
