#!/bin/bash
# train + test
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 main.py --data_path='../data/colon' --work_path='work/sa_mil/colon/0_img_ips256_bs4_lr1e-5_ep60' --model='SA_MIL' --epochs=60 --batch_size=4 --lr=1e-5 --input_size=256 --device_ids=0 --train --pretrain --test_num_pos=80 --save_all --test
# test only
python3 -W ignore main.py --data_path='../data/colon' --work_path='work/sa_mil/colon/0_img_ips256_bs4_lr1e-5_ep60' --model='SA_MIL' --input_size=256 --device='cuda:0' --test_num_pos=80 --test