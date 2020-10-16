import os
import glob

# 바꿀 이미지가 위치한 폴더
path = os.path.abspath("./data/image_inference_result02")
files = glob.glob(path + '/*')

for i, f in enumerate(files):
    os.rename(f, os.path.join(path, '{0:06d}'.format(i)+".jpg"))