#!/bin/bash

bash tools/dist_adv.sh  configs/retinanet/retinanet_x101_64x4d_fpn_1x_coco.py  checkpoints/retinanet_x101_64x4d_fpn_1x_coco_20200130-366f5af1.pth  3  --method difgsm  --show-dir data/adv_coco_8_5 --gen_adv_aug --eps 8 --alpha 2 --steps 5