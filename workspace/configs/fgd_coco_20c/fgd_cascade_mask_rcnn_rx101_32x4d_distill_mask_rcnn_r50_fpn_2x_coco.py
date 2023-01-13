_base_ = [
    './fgd_cascade_mask_rcnn_rx101_32x4d_distill_mask_rcnn_r50_fpn_1x_coco.py'
]

student_cfg = 'configs/mask_rcnn_coco_20c/mask_rcnn_r50_fpn_2x_coco.py'

lr_config = dict(step=[16, 22])
runner = dict(max_epochs=24)