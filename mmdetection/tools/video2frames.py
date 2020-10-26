import mmcv
import os

path = os.path.abspath("./data/KRRI_Video/00_Sinseondae_Pier_Lighting_Tower_CCTV_video/Daytime_3-day/CCTV-11_cctv11_0923_Daytime_CH01/CCTV-11_cctv11_0923_Daytime_CH01.avi")
video = mmcv.VideoReader(path)

print(len(video)) # get the total frame number
print(video.width, video.height, video.resolution, video.fps)

video.cvt2frames("/home/minjae/mjseong/mmdetection/data/person_test", start= 0, max_num= 2000)