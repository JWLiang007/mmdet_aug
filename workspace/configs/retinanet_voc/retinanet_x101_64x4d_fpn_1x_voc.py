_base_ = './retinanet_r50_fpn_1x_voc.py'
model = dict(
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://resnext101_64x4d',map_location='cpu')))


# ========
optimizer_config = dict(_delete_=True,grad_clip=dict(max_norm=35, norm_type=2))
batch_size = 2
data = dict(
    samples_per_gpu=batch_size,
)
auto_scale_lr = dict(enable=True)