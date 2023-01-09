_base_ = ["./fgd_retina_rx101_64x4d_distill_retina_r50_fpn_1x_coco.py"]

# model settings
find_unused_parameters = True
temp = 0.5
alpha_fgd = 0.001
beta_fgd = 0.0005
gamma_fgd = 0.0005
lambda_fgd = 0.000005
# adv loss settings
alpha_adv = 0.00001
loss_type = "mse"
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
            student_module="bbox_head.loss_cls",
            teacher_module="bbox_head.loss_cls",
            methods=[
                dict(
                    name="adv_dkd_loss",
                    loss_input_type="logit",
                    hook_type='input',
                    logit_filter="teacher",
                    img_type='adv',
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
        dict(
            student_module="neck.fpn_convs.4.conv",
            teacher_module="neck.fpn_convs.4.conv",
            methods=[
                dict(
                    name="fgd_loss_fpn_4",
                    loss_input_type="feature",
                    hook_type='output',
                    img_type='clean',
                    loss_param=fgd_param,
                ),
                dict(
                    name="adv_loss_fpn_4",
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
    dict(type="LoadImageFromFile", adv_img="data/adv_coco_20c_8_5/"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="InstanceAug", prob=0.3, subst_full=True, subst_stg="1"),
    dict(type="RandomFlip", flip_ratio=0.0),
    dict(
        type="RandomCrop",
        crop_size=(0.8, 0.8),
        crop_type="relative_range",
        adaptive=True,
        bbox_size=(32, 32),
        subst_stg="1",
    ),
    dict(type="Resize", img_scale=(1333, 800), keep_ratio=True),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "adv", "gt_bboxes", "gt_labels"]),
]

data = dict(train=dict(pipeline=train_pipeline), )

custom_hooks = [
    dict(type='NumClassCheckHook'),
    dict(type='SetRunModeHook'),
    dict(
        type='SetLossWeightHook',
        attrs={
            'adv_dkd_loss': ['alpha', 'beta'],
            'adv_loss_fpn_0': ['alpha_adv'],
            'adv_loss_fpn_1': ['alpha_adv'],
            'adv_loss_fpn_2': ['alpha_adv'],
            'adv_loss_fpn_3': ['alpha_adv'],
            'adv_loss_fpn_4': ['alpha_adv']
        })
]
