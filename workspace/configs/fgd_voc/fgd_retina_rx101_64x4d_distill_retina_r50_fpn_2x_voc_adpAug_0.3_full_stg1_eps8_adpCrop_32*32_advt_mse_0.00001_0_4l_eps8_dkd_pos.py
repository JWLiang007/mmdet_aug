_base_ = ["./fgd_retina_rx101_64x4d_distill_retina_r50_fpn_2x_voc.py"]


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
alpha_dkd = 1.0
beta_dkd = 0.25
temp_dkd = 1.0


fgd_loss = dict(
    type="FGDLoss",
    name="fgd_loss",
    student_channels=256,
    teacher_channels=256,
    temp=temp,
    alpha_fgd=alpha_fgd,
    beta_fgd=beta_fgd,
    gamma_fgd=gamma_fgd,
    lambda_fgd=lambda_fgd,
)
adv_loss = dict(
    type="AdvFeatureLoss",
    name="adv_loss",
    student_channels=256,
    teacher_channels=256,
    alpha_adv=alpha_adv,
    loss_type=loss_type,
)

distiller = dict(
    type="FGDDistiller",
    teacher_pretrained="checkpoints/retinanet_x101_voc_24.pth",
    init_student=True,
    distill_cfg=[
        # dict(
        # logit=[
        dict(
            student_module="bbox_head.loss_cls",
            teacher_module="bbox_head.loss_cls",
            output_hook=False,
            local_buffer=True,
            type="logit",
            methods=[
                dict(
                    type="DKDLoss",
                    name="dkd_loss",
                    alpha=alpha_dkd,
                    beta=beta_dkd,
                    temp=temp_dkd,
                ),
            ],
        ),
        # ],
        # feature=[
        dict(
            student_module="neck.fpn_convs.0.conv",
            teacher_module="neck.fpn_convs.0.conv",
            output_hook=True,
            local_buffer=False,
            type="feature",
            methods=[
                fgd_loss,
                adv_loss
                # dict(
                #     type="FGDLoss",
                #     name="loss_fgd_fpn_4",
                #     student_channels=256,
                #     teacher_channels=256,
                #     temp=temp,
                #     alpha_fgd=alpha_fgd,
                #     beta_fgd=beta_fgd,
                #     gamma_fgd=gamma_fgd,
                #     lambda_fgd=lambda_fgd,
                # ),
                # dict(
                #     type="AdvFeatureLoss",
                #     name="adv_loss_fgd_fpn_4",
                #     student_channels=256,
                #     teacher_channels=256,
                #     alpha_adv=alpha_adv,
                #     layer_idx=4,
                #     loss_type=loss_type,
                # ),
            ],
        ),
        dict(
            student_module="neck.fpn_convs.1.conv",
            teacher_module="neck.fpn_convs.1.conv",
            output_hook=True,
            local_buffer=False,
            type="feature",
            methods=[
                fgd_loss,
                adv_loss
                # dict(
                #     type="FGDLoss",
                #     name="loss_fgd_fpn_3",
                #     student_channels=256,
                #     teacher_channels=256,
                #     temp=temp,
                #     alpha_fgd=alpha_fgd,
                #     beta_fgd=beta_fgd,
                #     gamma_fgd=gamma_fgd,
                #     lambda_fgd=lambda_fgd,
                # ),
                # dict(
                #     type="AdvFeatureLoss",
                #     name="adv_loss_fgd_fpn_3",
                #     student_channels=256,
                #     teacher_channels=256,
                #     alpha_adv=alpha_adv,
                #     layer_idx=3,
                #     loss_type=loss_type,
                # ),
            ],
        ),
        dict(
            student_module="neck.fpn_convs.2.conv",
            teacher_module="neck.fpn_convs.2.conv",
            output_hook=True,
            local_buffer=False,
            type="feature",
            methods=[
                fgd_loss,
                adv_loss
                # dict(
                #     type="FGDLoss",
                #     name="loss_fgd_fpn_2",
                #     student_channels=256,
                #     teacher_channels=256,
                #     temp=temp,
                #     alpha_fgd=alpha_fgd,
                #     beta_fgd=beta_fgd,
                #     gamma_fgd=gamma_fgd,
                #     lambda_fgd=lambda_fgd,
                # ),
                # dict(
                #     type="AdvFeatureLoss",
                #     name="adv_loss_fgd_fpn_2",
                #     student_channels=256,
                #     teacher_channels=256,
                #     alpha_adv=alpha_adv,
                #     layer_idx=2,
                #     loss_type=loss_type,
                # ),
            ],
        ),
        dict(
            student_module="neck.fpn_convs.3.conv",
            teacher_module="neck.fpn_convs.3.conv",
            output_hook=True,
            local_buffer=False,
            type="feature",
            methods=[
                fgd_loss,
                adv_loss
                # dict(
                #     type="FGDLoss",
                #     name="loss_fgd_fpn_1",
                #     student_channels=256,
                #     teacher_channels=256,
                #     temp=temp,
                #     alpha_fgd=alpha_fgd,
                #     beta_fgd=beta_fgd,
                #     gamma_fgd=gamma_fgd,
                #     lambda_fgd=lambda_fgd,
                # ),
                # dict(
                #     type="AdvFeatureLoss",
                #     name="adv_loss_fgd_fpn_1",
                #     student_channels=256,
                #     teacher_channels=256,
                #     alpha_adv=alpha_adv,
                #     layer_idx=1,
                #     loss_type=loss_type,
                # ),
            ],
        ),
        dict(
            student_module="neck.fpn_convs.4.conv",
            teacher_module="neck.fpn_convs.4.conv",
            output_hook=True,
            local_buffer=False,
            type="feature",
            methods=[
                fgd_loss,
                adv_loss
                # dict(
                #     type="FGDLoss",
                #     name="loss_fgd_fpn_0",
                #     student_channels=256,
                #     teacher_channels=256,
                #     temp=temp,
                #     alpha_fgd=alpha_fgd,
                #     beta_fgd=beta_fgd,
                #     gamma_fgd=gamma_fgd,
                #     lambda_fgd=lambda_fgd,
                # ),
                # dict(
                #     type="AdvFeatureLoss",
                #     name="adv_loss_fgd_fpn_0",
                #     student_channels=256,
                #     teacher_channels=256,
                #     alpha_adv=alpha_adv,
                #     layer_idx=0,
                #     loss_type=loss_type,
                # ),
            ],
        ),
    ],
)


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
train_pipeline = [
    dict(type="LoadImageFromFile", adv_img="data/adv_voc_8_5/"),
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

data = dict(
    train=dict(pipeline=train_pipeline),
)
