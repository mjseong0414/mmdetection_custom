import mmcv
import os

path = os.path.abspath("./data/KRRI_Video/00_Sinseondae Pier_Lighting Tower CCTV video/Daytime 3-day/CCTV-11_cctv11_0923_Daytime_CH01/CCTV-11_cctv11_0923_Daytime_CH01.avi")
video = mmcv.VideoReader(path)

print(len(video)) # get the total frame number
print(video.width, video.height, video.resolution, video.fps)

video.cvt2frames("/home/minjae/mjseong/mmdetection/data/person_test", start= 216000, max_num= 234000)