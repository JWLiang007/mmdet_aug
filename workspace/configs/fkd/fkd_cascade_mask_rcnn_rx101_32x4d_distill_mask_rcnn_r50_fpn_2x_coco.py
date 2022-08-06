_base_ = [
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]
# model settings
find_unused_parameters=True
alpha_fkd=0.004
beta_fkd=0.00007
gamma_fkd=0.004
T_fkd=0.1
distiller = dict(
    type='FKDDistiller',
    is_two_stage = True,
    teacher_pretrained = 'checkpoints/cascade_mask_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco-e75f90c8.pth',
    # init_student = True,
    distill_cfg=[
                 dict(student_module='neck.fpn_convs.3.conv',
                      teacher_module='neck.fpn_convs.3.conv',
                      output_hook=True,
                      methods=[dict(type='FKDLoss',
                                    name='loss_fkd_fpn_3',
                                    alpha_fkd=alpha_fkd,
                                    beta_fkd=beta_fkd,
                                    gamma_fkd=gamma_fkd,
                                    T_fkd=T_fkd
                                    )
                               ]
                      ),
                 dict(student_module='neck.fpn_convs.2.conv',
                      teacher_module='neck.fpn_convs.2.conv',
                      output_hook=True,
                      methods=[dict(type='FKDLoss',
                                    name='loss_fkd_fpn_2',
                                    alpha_fkd=alpha_fkd,
                                    beta_fkd=beta_fkd,
                                    gamma_fkd=gamma_fkd,
                                    T_fkd=T_fkd
                                    )
                               ]
                      ),
                 dict(student_module='neck.fpn_convs.1.conv',
                      teacher_module='neck.fpn_convs.1.conv',
                      output_hook=True,
                      methods=[dict(type='FKDLoss',
                                    name='loss_fkd_fpn_1',
                                    alpha_fkd=alpha_fkd,
                                    beta_fkd=beta_fkd,
                                    gamma_fkd=gamma_fkd,
                                    T_fkd=T_fkd
                                    )
                               ]
                      ),
                 dict(student_module='neck.fpn_convs.0.conv',
                      teacher_module='neck.fpn_convs.0.conv',
                      output_hook=True,
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

student_cfg = 'configs/mask_rcnn/mask_rcnn_r50_fpn_2x_coco.py'
teacher_cfg = 'configs/dcn/cascade_mask_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco.py'
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,)
