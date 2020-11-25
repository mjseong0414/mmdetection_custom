'''
1. video2frames
2. implements inference
3. frame2video
'''

from mmdet.apis import init_detector, inference_detector, save_result_pyplot
import matplotlib.pyplot as plt
import mmcv
import argparse
import os
import cv2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help= 'test config file path')
    parser.add_argument('--checkpoint', help= 'checkpoint file')
    parser.add_argument('--rtsp_url', help= 'url path') # 예) rtsp://166.104.14.77:port/test
    parser.add_argument('--inference_path', help='inference image path')
    # parser.add_argument('--avi_path', help='avi path')
    args = parser.parse_args()

    config_file = args.config
    checkpoint_file = args.checkpoint
    rtsp_url = args.rtsp_url
    inference_path = os.path.abspath(args.inference_path) # inference 시킨 이미지들
    # avi_path = os.path.abspath(args.avi_path) # avi로 바꾼 이미지들


    model = init_detector(config_file, checkpoint_file, device= 'cuda:0')
    video = cv2.VideoCapture(rtsp_url)
    integer = 0
    while True:
        _, frame = video.read()
        result = inference_detector(model, frame)
        last_path = inference_path + "/{0:06d}".format(integer) + ".jpg"
        save_result_pyplot(model, frame, result, last_path)
        integer += 1
        # k = cv2.waitKey(1)
    import pdb; pdb.set_trace()

    # mmcv.frames2video(inference_path, avi_path, fps= 15)
    
if __name__=='__main__':
    main()