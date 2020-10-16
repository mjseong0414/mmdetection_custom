from pycocotools.coco import COCO
import requests
# 카테고리 아이디 -> 이미지 아이디추출, 이미지 아이디 -> 어노테이션 아이디 추출, 어노테이션 아이디 -> 어노테이션 로드
coco = COCO("/home/minjae/mjseong/mmdetection/data/coco/annotations/filtered_train.json")
catIds = coco.getCatIds(catNms = ['person', 'car', 'truck', 'cat', 'dog']) # catIds = [1, 3, 8, 17, 18]
imgIds = coco.getImgIds(catIds = 3) # catIds = ?, 물음표에 들어가는 카테고리 id를 포함한 train image의 id들이 모두 imgIds에 저장됨
annIds = coco.getAnnIds(imgIds)
annotation = coco.loadAnns(annIds)
import pdb; pdb.set_trace()