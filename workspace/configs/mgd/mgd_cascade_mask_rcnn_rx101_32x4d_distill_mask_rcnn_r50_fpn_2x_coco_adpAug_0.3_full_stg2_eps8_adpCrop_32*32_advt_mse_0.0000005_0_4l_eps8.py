_base_ = [
    './mgd_cascade_mask_rcnn_rx101_32x4d_distill_mask_rcnn_r50_fpn_2x_coco.py'
]
# model settings
find_unused_parameters=True
alpha_mgd=0.0000005
lambda_mgd=0.45
# adv loss settings
alpha_adv=0.0000005
loss_type='mse'

distiller = dict(
    type='MGDDistiller',
    teacher_pretrained = 'checkpoints/cascade_mask_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco-e75f90c8.pth',
    distill_cfg = [ dict(student_module = 'neck.fpn_convs.3.conv',
                         teacher_module = 'neck.fpn_convs.3.conv',
                         output_hook = True,
                         methods=[dict(type='MGDLoss',
                                       name='loss_mgd_fpn_3',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       alpha_mgd=alpha_mgd,
                                       lambda_mgd=lambda_mgd,
                                       ),
                                dict(type='AdvFeatureLoss',
                                       name='adv_loss_mgd_fpn_3',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       alpha_adv=alpha_adv,
                                        layer_idx=3,
                                        loss_type = loss_type,
                                       )
                                ]
                        ),
                    dict(student_module = 'neck.fpn_convs.2.conv',
                         teacher_module = 'neck.fpn_convs.2.conv',
                         output_hook = True,
                         methods=[dict(type='MGDLoss',
                                       name='loss_mgd_fpn_2',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       alpha_mgd=alpha_mgd,
                                       lambda_mgd=lambda_mgd,
                                       ),
                                dict(type='AdvFeatureLoss',
                                       name='adv_loss_mgd_fpn_2',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       alpha_adv=alpha_adv,
                                        layer_idx=2,
                                        loss_type = loss_type,
                                       )
                                ]
                        ),
                    dict(student_module = 'neck.fpn_convs.1.conv',
                         teacher_module = 'neck.fpn_convs.1.conv',
                         output_hook = True,
                         methods=[dict(type='MGDLoss',
                                       name='loss_mgd_fpn_1',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       alpha_mgd=alpha_mgd,
                                       lambda_mgd=lambda_mgd,
                                       ),
                                dict(type='AdvFeatureLoss',
                                       name='adv_loss_mgd_fpn_1',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       alpha_adv=alpha_adv,
                                        layer_idx=1,
                                        loss_type = loss_type,
                                       )
                                ]
                        ),
                    dict(student_module = 'neck.fpn_convs.0.conv',
                         teacher_module = 'neck.fpn_convs.0.conv',
                         output_hook = True,
                         methods=[dict(type='MGDLoss',
                                       name='loss_mgd_fpn_0',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       alpha_mgd=alpha_mgd,
                                       lambda_mgd=lambda_mgd,
                                       ),
                                dict(type='AdvFeatureLoss',
                                       name='adv_loss_mgd_fpn_0',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       alpha_adv=alpha_adv,
                                        layer_idx=0,
                                        loss_type = loss_type,
                                       )
                                ]
                        ),

                   ]
    )




img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile',adv_img='data/cmr_adv_coco_8_5/'),
    dict(type='LoadAnnotations', with_bbox=True ,with_mask=True),
    dict(type='InstanceAug',prob=0.3,subst_full=True,subst_stg='2'),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='RandomCrop', crop_size=(0.8, 0.8), crop_type='relative_range', adaptive=True, bbox_size=(32, 32),subst_stg='2'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'adv','gt_bboxes', 'gt_labels', 'gt_masks']),
]

data = dict(
    train=dict(
        pipeline=train_pipeline),
)