#!/bin/bash

./train_rt_lamp_detr.py --custom_data_path ./_data/rt-lamp-detr --augment --dropout 0.3 --batch_size=3 --num_queries=64 --lr_drop=200 --epochs=400 --set_cost_class 1 --set_cost_bbox 5 --set_cost_giou 2 --bbox_loss_coef=5 --giou_loss_coef=2 --eos_coef=0.1 --eval_test --early_stopping=400 --name "M015_AUGMENT_FINAL"
