_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/voc07_cocofmt.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]


# ======================
model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=20)
    )
)

auto_scale_lr = dict(enable=True)
batch_size = 2
data = dict(
    samples_per_gpu=batch_size,
)