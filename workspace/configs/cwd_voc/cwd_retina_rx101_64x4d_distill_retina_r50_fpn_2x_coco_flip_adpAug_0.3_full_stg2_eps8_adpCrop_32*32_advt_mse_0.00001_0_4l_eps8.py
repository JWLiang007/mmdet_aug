_base_ = [
    '../_base_/datasets/voc07_cocofmt.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]
# model settings
find_unused_parameters=True
weight=5.0
tau=1.0
# adv loss settings
alpha_adv=0.00001
loss_type='mse'

distiller = dict(
    type='CWDDistiller',
    teacher_pretrained = 'checkpoints/retinanet_x101_voc_24.pth',
    init_student = True,
    distill_cfg = [ dict(student_module = 'neck.fpn_convs.4.conv',
                         teacher_module = 'neck.fpn_convs.4.conv',
                         output_hook = True,
                         methods=[dict(type='CWDLoss',
                                       name='loss_cwd_fpn_4',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       tau = tau,
                                       weight =weight,
                                       ),
                                       dict(type='AdvFeatureLoss',
                                       name='adv_loss_cwd_fpn_4',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       alpha_adv=alpha_adv,
                                        layer_idx=4,
                                        loss_type = loss_type,
                                       )
                                ]
                        ), 
                    dict(student_module = 'neck.fpn_convs.3.conv',
                         teacher_module = 'neck.fpn_convs.3.conv',
                         output_hook = True,
                         methods=[dict(type='CWDLoss',
                                       name='loss_cwd_fpn_3',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       tau = tau,
                                       weight =weight,
                                       ),
                                       dict(type='AdvFeatureLoss',
                                       name='adv_loss_cwd_fpn_3',
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

                         methods=[dict(type='CWDLoss',
                                       name='loss_cwd_fpn_2',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       tau = tau,
                                       weight =weight,
                                       ),
                                dict(type='AdvFeatureLoss',
                                       name='adv_loss_cwd_fpn_2',
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

                         methods=[dict(type='CWDLoss',
                                       tau = tau,
                                       name='loss_cwd_fpn_1',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       weight =weight,
                                       ),
                                dict(type='AdvFeatureLoss',
                                       name='adv_loss_cwd_fpn_1',
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

                         methods=[dict(type='CWDLoss',
                                       tau = tau,
                                       name='loss_cwd_fpn_0',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       weight =weight,
                                       ),
                                dict(type='AdvFeatureLoss',
                                       name='adv_loss_cwd_fpn_0',
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

student_cfg = 'configs/retinanet_voc/retinanet_r50_fpn_2x_voc.py'
teacher_cfg = 'configs/retinanet_voc/retinanet_x101_64x4d_fpn_1x_voc.py'
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile',adv_img='data/adv_voc_8_5/'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='InstanceAug',prob=0.3,subst_full=True,subst_stg='2'),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='RandomCrop', crop_size=(0.8, 0.8), crop_type='relative_range', adaptive=True, bbox_size=(32, 32),subst_stg='2'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'adv','gt_bboxes', 'gt_labels']),
]

data = dict(
    train=dict(
        pipeline=train_pipeline),
)
