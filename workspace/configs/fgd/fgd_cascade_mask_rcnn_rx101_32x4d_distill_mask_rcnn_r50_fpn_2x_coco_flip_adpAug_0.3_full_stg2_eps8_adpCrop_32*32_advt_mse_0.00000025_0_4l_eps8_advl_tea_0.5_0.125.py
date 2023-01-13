_base_ = [
    './fgd_cascade_mask_rcnn_rx101_32x4d_distill_mask_rcnn_r50_fpn_2x_coco.py'
]
# model settings
find_unused_parameters = True
temp = 0.5
alpha_fgd = 0.00005
beta_fgd = 0.000025
gamma_fgd = 0.00005
lambda_fgd = 0.0000005
# adv loss settings
alpha_adv = 0.00000025
loss_type = 'mse'
# dkd loss settings
alpha_dkd = 0.5
beta_dkd = 0.125
temp_dkd = 1.0

fgd_param = dict(
    type="FGDLoss",
    # name="fgd_loss",
    student_channels=256,
    teacher_channels=256,
    temp=temp,
    alpha_fgd=alpha_fgd,
    beta_fgd=beta_fgd,
    gamma_fgd=gamma_fgd,
    lambda_fgd=lambda_fgd,
)

adv_feat_param = dict(
    type="AdvFeatureLoss",
    # name="adv_loss",
    student_channels=256,
    teacher_channels=256,
    alpha_adv=alpha_adv,
    loss_type=loss_type,
)

adv_dkd_param = dict(
    type="DKDLoss",
    # name="dkd_loss",
    alpha=alpha_dkd,
    beta=beta_dkd,
    temp=temp_dkd,
)

distiller = dict(
    # type="FGDDistiller",
    # teacher_pretrained="checkpoints/retinanet_x101_voc_24.pth",
    # init_student=True,
    distill_cfg=[
        dict(
            student_module="roi_head.bbox_head.loss_cls",
            teacher_module="roi_head.bbox_head.2.loss_cls",

            methods=[
                dict(
                    name="adv_dkd_loss",
                    loss_input_type="logit",
                    hook_type = 'input',
                    logit_filter="teacher",
                    img_type = 'adv',
                    loss_param=adv_dkd_param,
                ),
            ],
        ),
        dict(
            student_module="neck.fpn_convs.0.conv",
            teacher_module="neck.fpn_convs.0.conv",
            methods=[
                dict(
                    name="fgd_loss_fpn_0",
                    loss_input_type="feature",
                    hook_type='output',
                    img_type='clean',
                    loss_param=fgd_param,
                ),
                dict(
                    name="adv_loss_fpn_0",
                    loss_input_type="feature",
                    hook_type='output',
                    img_type='adv',
                    loss_param=adv_feat_param,
                ),
            ],
        ),
        dict(
            student_module="neck.fpn_convs.1.conv",
            teacher_module="neck.fpn_convs.1.conv",
            methods=[
                dict(
                    name="fgd_loss_fpn_1",
                    loss_input_type="feature",
                    hook_type='output',
                    img_type='clean',
                    loss_param=fgd_param,
                ),
                dict(
                    name="adv_loss_fpn_1",
                    loss_input_type="feature",
                    hook_type='output',
                    img_type='adv',
                    loss_param=adv_feat_param,
                ),
            ],
        ),
        dict(
            student_module="neck.fpn_convs.2.conv",
            teacher_module="neck.fpn_convs.2.conv",
            methods=[
                dict(
                    name="fgd_loss_fpn_2",
                    loss_input_type="feature",
                    hook_type='output',
                    img_type='clean',
                    loss_param=fgd_param,
                ),
                dict(
                    name="adv_loss_fpn_2",
                    loss_input_type="feature",
                    hook_type='output',
                    img_type='adv',
                    loss_param=adv_feat_param,
                ),
            ],
        ),
        dict(
            student_module="neck.fpn_convs.3.conv",
            teacher_module="neck.fpn_convs.3.conv",
            methods=[
                dict(
                    name="fgd_loss_fpn_3",
                    loss_input_type="feature",
                    hook_type='output',
                    img_type='clean',
                    loss_param=fgd_param,
                ),
                dict(
                    name="adv_loss_fpn_3",
                    loss_input_type="feature",
                    hook_type='output',
                    img_type='adv',
                    loss_param=adv_feat_param,
                ),
            ],
        ),
    ], )

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', adv_img='data/cmr_adv_coco_8_5/'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='InstanceAug', prob=0.3, subst_full=True, subst_stg='1'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='RandomCrop',
        crop_size=(0.8, 0.8),
        crop_type='relative_range',
        adaptive=True,
        bbox_size=(32, 32),
        subst_stg='1'),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'adv', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]

data = dict(train=dict(pipeline=train_pipeline), )
