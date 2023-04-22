_base_ = [
    '../_base_/datasets/voc0712.py',
   #  '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py'
]
# model settings
find_unused_parameters=True
temp=0.5
alpha_fkd=7e-5 * 6
beta_fkd=4e-3 * 6
gamma_fkd=7e-5 * 6
# lambda_fkd=0.000005
distiller = dict(
    type='FKDDistiller',
    teacher_pretrained = 'checkpoints/retinanet_x101_64x4d_fpn_1x_voc.pth',
    # init_student = True,
    distill_cfg = [ dict(student_module = 'neck.fpn_convs.4.conv',
                         teacher_module = 'neck.fpn_convs.4.conv',
                         output_hook = True,
                         methods=[dict(type='FKDLoss',
                                       name='loss_fkd_fpn_4',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       temp = temp,
                                       alpha_fkd=alpha_fkd,
                                       beta_fkd=beta_fkd,
                                       gamma_fkd=gamma_fkd,
                                       layer_idx=4,
                                    #    lambda_fkd=lambda_fkd,
                                       )
                                ]
                        ),
                    dict(student_module = 'neck.fpn_convs.3.conv',
                         teacher_module = 'neck.fpn_convs.3.conv',
                         output_hook = True,
                         methods=[dict(type='FKDLoss',
                                       name='loss_fkd_fpn_3',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       temp = temp,
                                       alpha_fkd=alpha_fkd,
                                       beta_fkd=beta_fkd,
                                       gamma_fkd=gamma_fkd,
                                       layer_idx=3,
                                    #    lambda_fkd=lambda_fkd,
                                       )
                                ]
                        ),
                    dict(student_module = 'neck.fpn_convs.2.conv',
                         teacher_module = 'neck.fpn_convs.2.conv',
                         output_hook = True,
                         methods=[dict(type='FKDLoss',
                                       name='loss_fkd_fpn_2',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       temp = temp,
                                       alpha_fkd=alpha_fkd,
                                       beta_fkd=beta_fkd,
                                       gamma_fkd=gamma_fkd,
                                       layer_idx=2,
                                    #    lambda_fkd=lambda_fkd,
                                       )
                                ]
                        ),
                    dict(student_module = 'neck.fpn_convs.1.conv',
                         teacher_module = 'neck.fpn_convs.1.conv',
                         output_hook = True,
                         methods=[dict(type='FKDLoss',
                                       name='loss_fkd_fpn_1',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       temp = temp,
                                       alpha_fkd=alpha_fkd,
                                       beta_fkd=beta_fkd,
                                       gamma_fkd=gamma_fkd,
                                       layer_idx=1,
                                    #    lambda_fkd=lambda_fkd,
                                       )
                                ]
                        ),
                    dict(student_module = 'neck.fpn_convs.0.conv',
                         teacher_module = 'neck.fpn_convs.0.conv',
                         output_hook = True,
                         methods=[dict(type='FKDLoss',
                                       name='loss_fkd_fpn_0',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       temp = temp,
                                       alpha_fkd=alpha_fkd,
                                       beta_fkd=beta_fkd,
                                       gamma_fkd=gamma_fkd,
                                       layer_idx=0,
                                    #    lambda_fkd=lambda_fkd,
                                       )
                                ]
                        ),

                   ]
    )

student_cfg = 'configs/pascal_voc/retinanet_r50_fpn_1x_voc0712.py'
teacher_cfg = 'configs/pascal_voc/retinanet_x101_64x4d_fpn_1x_voc0712.py'
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=16,)


# learning policy
# actual epoch = 3 * 3 = 9
lr_config = dict(policy='step', step=[3])
# runtime settings
runner = dict(
    type='EpochBasedRunner', max_epochs=4)  # actual epoch = 4 * 3 = 12