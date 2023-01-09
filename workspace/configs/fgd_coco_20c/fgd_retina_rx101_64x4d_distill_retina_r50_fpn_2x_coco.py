_base_ = [
    './fgd_retina_rx101_64x4d_distill_retina_r50_fpn_1x_coco.py',
]

lr_config = dict(step=[16, 22])
runner = dict(max_epochs=24)

student_cfg = 'configs/retinanet_coco_20c/retinanet_r50_fpn_2x_coco.py'