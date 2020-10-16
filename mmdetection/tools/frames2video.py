import mmcv
import os

path = os.path.abspath("./data/image_inference_result02/")
mmcv.frames2video(path, "/home/minjae/mjseong/mmdetection/data/test02.avi", fps= 15)
# mmcv.frames2video("/home/minjae/mjseong/mmdetection/data/KRRI_Video_cvt2frames_6000", "/home/minjae/mjseong/mmdetection/data/6000test.avi", fps= 20)