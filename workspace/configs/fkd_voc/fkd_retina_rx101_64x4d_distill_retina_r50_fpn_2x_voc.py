_base_ = [
    '../_base_/datasets/voc07_cocofmt.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]
# model settings
find_unused_parameters=True
alpha_fkd=0.0004
beta_fkd=0.02
gamma_fkd=0.0004
T_fkd=0.5
model = dict()
distiller = dict(
    type='FKDDistiller',
    teacher_pretrained = 'checkpoints/retinanet_x101_voc_24.pth',
    # init_student = True,
    distill_cfg = [ dict(student_module = 'neck.fpn_convs.4.conv',
                         teacher_module = 'neck.fpn_convs.4.conv',
                         output_hook = True,
                         methods=[dict(type='FKDLoss',
                                       name='loss_fkd_fpn_4',
                                       alpha_fkd=alpha_fkd,
                                       beta_fkd=beta_fkd,
                                       gamma_fkd=gamma_fkd,
                                       T_fkd=T_fkd
                                       )
                                ]
                        ),
                    dict(student_module = 'neck.fpn_convs.3.conv',
                         teacher_module = 'neck.fpn_convs.3.conv',
                         output_hook = True,
                         methods=[dict(type='FKDLoss',
                                       name='loss_fkd_fpn_3',
                                       alpha_fkd=alpha_fkd,
                                       beta_fkd=beta_fkd,
                                       gamma_fkd=gamma_fkd,
                                       T_fkd=T_fkd
                                       )
                                ]
                        ),
                    dict(student_module = 'neck.fpn_convs.2.conv',
                         teacher_module = 'neck.fpn_convs.2.conv',
                         output_hook = True,
                         methods=[dict(type='FKDLoss',
                                       name='loss_fkd_fpn_2',
                                       alpha_fkd=alpha_fkd,
                                       beta_fkd=beta_fkd,
                                       gamma_fkd=gamma_fkd,
                                       T_fkd=T_fkd
                                       )
                                ]
                        ),
                    dict(student_module = 'neck.fpn_convs.1.conv',
                         teacher_module = 'neck.fpn_convs.1.conv',
                         output_hook = True,
                         methods=[dict(type='FKDLoss',
                                       name='loss_fkd_fpn_1',
                                       alpha_fkd=alpha_fkd,
                                       beta_fkd=beta_fkd,
                                       gamma_fkd=gamma_fkd,
                                       T_fkd=T_fkd
                                       )
                                ]
                        ),
                    dict(student_module = 'neck.fpn_convs.0.conv',
                         teacher_module = 'neck.fpn_convs.0.conv',
                         output_hook = True,
                         methods=[dict(type='FKDLoss',
                                       name='loss_fkd_fpn_0',
                                       alpha_fkd=alpha_fkd,
                                       beta_fkd=beta_fkd,
                                       gamma_fkd=gamma_fkd,
                                       T_fkd=T_fkd
                                       )
                                ]
                        ),

                   ]
    )

student_cfg = 'configs/retinanet_voc/retinanet_r50_fpn_2x_voc.py'
teacher_cfg = 'configs/retinanet_voc/retinanet_x101_64x4d_fpn_1x_voc.py'

# ===================
batch_size = 4
optimizer = dict( lr=0.01 / (16/batch_size))
optimizer_config = dict(_delete_=True,grad_clip=dict(max_norm=35, norm_type=2))
data = dict(
    samples_per_gpu=batch_size,
)
evaluation = dict(interval=1)
checkpoint_config= dict(interval=1)