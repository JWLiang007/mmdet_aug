_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# model settings
find_unused_parameters = True
alpha_mgd = 0.0000005
lambda_mgd = 0.45

mgd_param = dict(
    type="MGDLoss",
    # name="mgd_loss",
    student_channels=256,
    teacher_channels=256,
    alpha_mgd=alpha_mgd,
    lambda_mgd=lambda_mgd,
)
distiller = dict(
    type="FGDDistiller",
    teacher_pretrained="checkpoints/retinanet_x101_coco_20c_12.pth",
    init_student=True,
    distill_cfg=[
        dict(
            student_module="neck.fpn_convs.0.conv",
            teacher_module="neck.fpn_convs.0.conv",
            methods=[
                dict(
                    name="mgd_loss_fpn_0",
                    loss_input_type="feature",
                    hook_type='output',
                    img_type='clean',
                    loss_param=mgd_param,
                ),
            ],
        ),
        dict(
            student_module="neck.fpn_convs.1.conv",
            teacher_module="neck.fpn_convs.1.conv",
            methods=[
                dict(
                    name="mgd_loss_fpn_1",
                    loss_input_type="feature",
                    hook_type='output',
                    img_type='clean',
                    loss_param=mgd_param,
                ),
            ],
        ),
        dict(
            student_module="neck.fpn_convs.2.conv",
            teacher_module="neck.fpn_convs.2.conv",
            methods=[
                dict(
                    name="mgd_loss_fpn_2",
                    loss_input_type="feature",
                    hook_type='output',
                    img_type='clean',
                    loss_param=mgd_param,
                ),
            ],
        ),
        dict(
            student_module="neck.fpn_convs.3.conv",
            teacher_module="neck.fpn_convs.3.conv",
            methods=[
                dict(
                    name="mgd_loss_fpn_3",
                    loss_input_type="feature",
                    hook_type='output',
                    img_type='clean',
                    loss_param=mgd_param,
                ),
            ],
        ),
        dict(
            student_module="neck.fpn_convs.4.conv",
            teacher_module="neck.fpn_convs.4.conv",
            methods=[
                dict(
                    name="mgd_loss_fpn_4",
                    loss_input_type="feature",
                    hook_type='output',
                    img_type='clean',
                    loss_param=mgd_param,
                ),
            ],
        ),
    ])

student_cfg = 'configs/retinanet_coco_20c/retinanet_r50_fpn_1x_coco.py'
teacher_cfg = 'configs/retinanet_coco_20c/retinanet_x101_64x4d_fpn_1x_coco.py'
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
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
