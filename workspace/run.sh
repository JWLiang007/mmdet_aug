#!/bin/bash

#  python ../tools/train.py configs/cascade_rcnn_voc/cascade_rcnn_x101_64x4d_fpn_1x_voc.py 
#  python ../tools/train.py configs/faster_rcnn_voc/faster_rcnn_r50_fpn_2x_voc.py
#  python ../tools/train.py configs/faster_rcnn_voc/faster_rcnn_r50_fpn_2x_voc_psd.py
#  python ../tools/train.py configs/faster_rcnn_voc/faster_rcnn_x101_64x4d_fpn_2x_voc.py --resume-from  work_dirs/faster_rcnn_x101_64x4d_fpn_2x_voc/latest.pth
 python ../tools/train.py configs/cwd_voc/cwd_cascade_rcnn_rx101_64x4d_distill_faster_rcnn_r50_fpn_2x_voc.py

