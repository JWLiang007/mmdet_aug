_base_ = [
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# model settings
find_unused_parameters = True
temp = 0.5
alpha_fgd = 0.00005
beta_fgd = 0.000025
gamma_fgd = 0.00005
lambda_fgd = 0.0000005
distiller = dict(
    type='FGDDistiller',
    teacher_pretrained=
    'checkpoints/cascade_mask_rcnn_x101_coco_20c_12.pth',
    distill_cfg=[
        dict(
            student_module='neck.fpn_convs.3.conv',
            teacher_module='neck.fpn_convs.3.conv',
            output_hook=True,
            methods=[
                dict(
                    type='FGDLoss',
                    name='loss_fgd_fpn_3',
                    student_channels=256,
                    teacher_channels=256,
                    temp=temp,
                    alpha_fgd=alpha_fgd,
                    beta_fgd=beta_fgd,
                    gamma_fgd=gamma_fgd,
                    lambda_fgd=lambda_fgd,
                )
            ]),
        dict(
            student_module='neck.fpn_convs.2.conv',
            teacher_module='neck.fpn_convs.2.conv',
            output_hook=True,
            methods=[
                dict(
                    type='FGDLoss',
                    name='loss_fgd_fpn_2',
                    student_channels=256,
                    teacher_channels=256,
                    temp=temp,
                    alpha_fgd=alpha_fgd,
                    beta_fgd=beta_fgd,
                    gamma_fgd=gamma_fgd,
                    lambda_fgd=lambda_fgd,
                )
            ]),
        dict(
            student_module='neck.fpn_convs.1.conv',
            teacher_module='neck.fpn_convs.1.conv',
            output_hook=True,
            methods=[
                dict(
                    type='FGDLoss',
                    name='loss_fgd_fpn_1',
                    student_channels=256,
                    teacher_channels=256,
                    temp=temp,
                    alpha_fgd=alpha_fgd,
                    beta_fgd=beta_fgd,
                    gamma_fgd=gamma_fgd,
                    lambda_fgd=lambda_fgd,
                )
            ]),
        dict(
            student_module='neck.fpn_convs.0.conv',
            teacher_module='neck.fpn_convs.0.conv',
            output_hook=True,
            methods=[
                dict(
                    type='FGDLoss',
                    name='loss_fgd_fpn_0',
                    student_channels=256,
                    teacher_channels=256,
                    temp=temp,
                    alpha_fgd=alpha_fgd,
                    beta_fgd=beta_fgd,
                    gamma_fgd=gamma_fgd,
                    lambda_fgd=lambda_fgd,
                )
            ]),
    ])

student_cfg = 'configs/mask_rcnn_coco_20c/mask_rcnn_r50_fpn_1x_coco.py'
teacher_cfg = 'configs/dcn_coco_20c/cascade_mask_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco.py'
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
auto_scale_lr = dict(enable=True)

CLASSES = ('bicycle', 'train', 'fire hydrant', 'stop sign', 'dog', 'bear',
           'giraffe', 'snowboard', 'baseball bat', 'bottle', 'sandwich',
           'broccoli', 'carrot', 'remote', 'cell phone', 'microwave',
           'toaster', 'sink', 'clock', 'vase')
data_root = 'data/coco/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        ann_file=data_root + 'annotations/instances_train2017_20c.json',
        img_prefix=data_root + 'train2017/',
        classes=CLASSES),
    val=dict(
        ann_file=data_root + 'annotations/instances_val2017_20c.json',
        img_prefix=data_root + 'val2017/',
        classes=CLASSES),
    test=dict(
        ann_file=data_root + 'annotations/instances_val2017_20c.json',
        img_prefix=data_root + 'val2017/',
        classes=CLASSES))

custom_hooks = [dict(type='NumClassCheckHook'), dict(type='SetRunModeHook')]
