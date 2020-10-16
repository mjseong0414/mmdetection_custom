'''
1. video2frames
2. implements inference
3. frame2video
'''

from mmdet.apis import init_detector, inference_detector, save_result_pyplot
import mmcv
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help= 'test config file path')
    parser.add_argument('--checkpoint', help= 'checkpoint file')
    parser.add_argument('--video_path', help= 'video_path')
    parser.add_argument('--frame_path', help= 'frame_path')
    parser.add_argument('--inference_path', help='inference image path')
    parser.add_argument('--avi_path', help='avi path')
    args = parser.parse_args()

    config_file = args.config
    checkpoint_file = args.checkpoint
    video_path = os.path.abspath(args.video_path) # ex) video_path = ./data/CCTV-11_cctv11_0923_Daytime_CH01.avi
    frame_path = os.path.abspath(args.frame_path) # frame으로 바꾼 이미지들, inference를 시킬 이미지들.
    inference_path = os.path.abspath(args.inference_path) # inference 시킨 이미지들
    avi_path = os.path.abspath(args.avi_path) # avi로 바꾼 이미지들

    ######### 1. video2frames
    video = mmcv.VideoReader(video_path)
    
    print(len(video)) # get the total frame number
    # print(video.width, video.height, video.resolution, video.fps)
    # ex) start = 150(기존 영상의 10초부터 시작), max_num = 785 -------------------> 기존 영상의 19초부터 
    video.cvt2frames(frame_path, start= 6200, max_num= 0) # frame을 (H, W, C) = (960, 1280, 3)으로 만들게 됨, 0~600 frame이 40초 영상으로 변환됨
    # mmcv.frames2video(frame_path, avi_path, fps= 15)
    
    ######### 2. implements inference
    model = init_detector(config_file, checkpoint_file, device= 'cuda:0')
    images = sorted(os.listdir(frame_path))
    integer = 0

    # images에 있는 image를 하나씩 뽑아가면서 inference 시켜야함
    for imgs in images: 
        img_for_inference = frame_path + "/" + imgs
        result = inference_detector(model, img_for_inference)
        last_path = inference_path + "/{0:06d}".format(integer) + ".jpg"
        save_result_pyplot(model, img_for_inference, result, last_path)
        integer += 1

    ########## 3. frame2video
    mmcv.frames2video(inference_path, avi_path, fps= 15)
    
if __name__=='__main__':
    main()