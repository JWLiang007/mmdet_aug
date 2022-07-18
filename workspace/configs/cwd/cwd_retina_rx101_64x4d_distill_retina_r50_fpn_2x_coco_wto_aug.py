_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]
# model settings
find_unused_parameters=True
weight=5.0
tau=1.0
distiller = dict(
    type='CWDDistiller',
    teacher_pretrained = 'checkpoints/retinanet_x101_64x4d_fpn_1x_coco_20200130-366f5af1.pth',
    distill_cfg = [ dict(student_module = 'neck.fpn_convs.3.conv',
                         teacher_module = 'neck.fpn_convs.3.conv',
                         output_hook = True,
                         methods=[dict(type='CWDLoss',
                                       name='loss_cw_fpn_3',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       tau = tau,
                                       weight =weight,
                                       )
                                ]
                        ),
                    dict(student_module = 'neck.fpn_convs.2.conv',
                         teacher_module = 'neck.fpn_convs.2.conv',
                         output_hook = True,

                         methods=[dict(type='CWDLoss',
                                       tau = tau,
                                       name='loss_cw_fpn_2',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       weight =weight,
                                       )
                                ]
                        ),
                    dict(student_module = 'neck.fpn_convs.1.conv',
                         teacher_module = 'neck.fpn_convs.1.conv',
                         output_hook = True,

                         methods=[dict(type='CWDLoss',
                                       tau = tau,
                                       name='loss_cw_fpn_1',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       weight =weight,
                                       )
                                ]
                        ),
                    dict(student_module = 'neck.fpn_convs.0.conv',
                         teacher_module = 'neck.fpn_convs.0.conv',
                         output_hook = True,

                         methods=[dict(type='CWDLoss',
                                       tau = tau,
                                       name='loss_cw_fpn_0',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       weight =weight,
                                       )
                                ]
                        ),


                   ]
    )

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]


student_cfg = 'configs/retinanet/retinanet_r50_fpn_2x_coco.py'
teacher_cfg = 'configs/retinanet/retinanet_x101_64x4d_fpn_1x_coco.py'
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        pipeline=train_pipeline
    )
)
