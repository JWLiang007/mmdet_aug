_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]


# ======================
model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=20),
        mask_head=dict(
            num_classes=20)
    )
)


auto_scale_lr = dict(enable=True)

CLASSES = ('bicycle', 'train', 'fire hydrant', 'stop sign', 'dog', 'bear', 'giraffe', 'snowboard', 'baseball bat', 'bottle', 'sandwich', 'broccoli', 'carrot', 'remote', 'cell phone', 'microwave', 'toaster', 'sink', 'clock', 'vase')

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