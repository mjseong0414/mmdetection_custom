# coco dataset에서 카테고리 id를 포함하는 이미지를 모두 가져오는 코드
from pycocotools.coco import COCO
import requests

coco = COCO("/home/minjae/mjseong/mmdetection/data/coco/annotations/instances_train2017.json")
catIds = coco.getCatIds(catNms = ['person', 'car', 'truck', 'cat', 'dog']) # catIds = [1, 3, 8, 17, 18]
# lists = [1, 3, 8, 17, 18]
lists = [1, 3, 8, 17, 18]
for i in lists:
    print("**************************"+str(i)+"*******************************")
    imgIds = coco.getImgIds(catIds = i) # catIds = ?, 물음표에 들어가는 카테고리 id를 포함한 train image의 id들이 모두 imgIds에 저장됨
    imgIds_sorted = sorted(imgIds)
    images = coco.loadImgs(imgIds_sorted)
    import pdb; pdb.set_trace()
    for im in images:
        print(im['file_name'])
        img_data = requests.get(im['coco_url']).content
        with open("/home/minjae/mjseong/mmdetection/data/coco_per_class/truck/" + im["file_name"], "wb") as handler:
            handler.write(img_data)

print("FINISH")
# person = 64115장
# car = 12251장
# truck = 6127장
# cat = 4114장
# dog = 4385장
# 12661