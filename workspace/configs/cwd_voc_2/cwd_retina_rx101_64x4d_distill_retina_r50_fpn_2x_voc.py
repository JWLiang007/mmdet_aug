_base_ = [
    '../_base_/datasets/voc07_cocofmt.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]
# model settings
find_unused_parameters=True
weight=5.0
tau=1.0
distiller = dict(
    type='CWDDistiller',
    teacher_pretrained = 'checkpoints/retinanet_x101_voc_24.pth',
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

student_cfg = 'configs/retinanet_voc/retinanet_r50_fpn_2x_voc.py'
teacher_cfg = 'configs/retinanet_voc/retinanet_x101_64x4d_fpn_1x_voc.py'

# ===================
batch_size = 2
optimizer = dict( lr=0.01 / (8/batch_size))
optimizer_config = dict(_delete_=True,grad_clip=dict(max_norm=35, norm_type=2))
data = dict(
    samples_per_gpu=batch_size,
)
evaluation = dict(interval=24)
checkpoint_config= dict(interval=1,max_keep_ckpts=1)