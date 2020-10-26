from mmdet.apis import init_detector, inference_detector, save_result_pyplot
import mmcv
import argparse
import os
import torch
import time
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help= 'test config file path')
    parser.add_argument('--checkpoint', help= 'checkpoint file')
    parser.add_argument('--frame_path', help= 'frame_path')
    parser.add_argument(
        '--log-interval', default=50, help='interval of logging')
    args = parser.parse_args()

    config_file = args.config
    checkpoint_file = args.checkpoint
    frame_path = os.path.abspath(args.frame_path) # frame으로 바꾼 이미지들, inference를 시킬 이미지들.
    
    ######### implements inference
    model = init_detector(config_file, checkpoint_file, device= 'cuda:0')
    images = sorted(os.listdir(frame_path))
    num_warmup = 5
    # pure_inf_time = 0

    # images에 있는 image를 하나씩 뽑아가면서 inference 시켜야함
    starter, ender = torch.cuda.Event(enable_timing= True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros((len(images),1))


    # measure performance
    with torch.no_grad():
        for i, imgs in enumerate(images):
            starter.record()
            if i
            ender.record()
            # wait for gpu sync
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[i] = curr_time
    
    # mean_syn = np.sum(timings) / len(images)
    mean_syn = len(images) / np.sum(timings)
    print(f'fps = {mean_syn} img / s')


    # for i, imgs in enumerate(images):
    #     torch.cuda.async()
    #     start_time = time.perf_counter()

    #     img_for_inference = frame_path + "/" + imgs
    #     inference_detector(model, img_for_inference)

    #     torch.cuda.synchronize()
    #     elapsed = time.perf_counter() - start_time
        
    #     if i >= num_warmup:
    #         pure_inf_time += elapsed
    #         if (i + 1) % args.log_interval == 0:
    #             fps = (i + 1 - num_warmup) / pure_inf_time
    #             print(f'Done image [{i + 1:<3}/ 2000], fps: {fps:.1f} img / s')
        
    #     if (i + 1) == 2000:
    #         pure_inf_time += elapsed
    #         fps = (i + 1 - num_warmup) / pure_inf_time
    #         print(f'Overall fps: {fps:.1f} img / s')
    #         break
    
if __name__=='__main__':
    main()