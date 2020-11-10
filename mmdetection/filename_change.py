import os
import glob

# 바꿀 이미지가 위치한 폴더
path = os.path.abspath("./data/krri/new_data/images_name_change")
files = glob.glob(path + '/*')

for i, f in enumerate(files):
    os.rename(f, os.path.join(path, '{0:06d}'.format(i)+".jpg"))