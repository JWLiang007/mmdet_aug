_base_ = '../cascade_rcnn_voc/cascade_mask_rcnn_x101_32x4d_fpn_1x_voc.py'
model = dict(
    backbone=dict(
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)))


# ======================
auto_scale_lr = dict(enable=True)
batch_size = 4
data = dict(
    samples_per_gpu=batch_size,
)