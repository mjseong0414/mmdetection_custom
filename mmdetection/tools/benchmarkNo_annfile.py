import argparse
import time
import os

import torch
from mmcv import Config
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model

from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector

## 임의 추가
from mmdet.apis import init_detector, inference_detector, save_result_pyplot
###############

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet benchmark a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--log-interval', default=50, help='interval of logging')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    frame_path = "./data/coco/val2017/"
    # frame_path = "./data/person_test/"
    # build the model and load checkpoint
    model = init_detector(args.config, args.checkpoint, device= 'cuda:0')
    images = sorted(os.listdir(frame_path))

    # the first several iterations may be very slow so skip them
    num_warmup = 5
    pure_inf_time = 0

    # benchmark with 2000 image and take the average
    for i, data in enumerate(images):
        
        img_for_inference = frame_path + "/" + data
        # inference.py의 inference_detector함수에서 elapsed하는 코드 추가해줘야 함
        result, elapsed = inference_detector(model, img_for_inference)

        if i >= num_warmup:
            pure_inf_time += elapsed
            if (i + 1) % args.log_interval == 0:
                fps = (i + 1 - num_warmup) / pure_inf_time
                print(f'Done image [{i + 1:<3}/ 2000], fps: {fps:.1f} img / s')

        if (i + 1) == 2000:
            pure_inf_time += elapsed
            fps = (i + 1 - num_warmup) / pure_inf_time
            print(f'Overall fps: {fps:.1f} img / s')
            break


if __name__ == '__main__':
    main()
