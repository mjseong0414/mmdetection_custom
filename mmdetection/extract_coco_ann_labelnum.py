from pycocotools.coco import COCO
import requests, os
import json
import shutil

'''

'''

coco = COCO(os.path.abspath('data/coco/annotations/instances_train2017.json'))
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

old_cat_Id = coco.getCatIds()
new_cat_Id = [1,2,3,4,5]

#initialize
catIds = []
imgIds = []
person_img_num = 10001
want_class = ['person','car','truck']

for classes in want_class:
    catIds += coco.getCatIds(catNms=[classes])
print(catIds)

for i in range(len(want_class)):
    if catIds[i] == 1:
        person_ImgIds = coco.getImgIds(catIds=catIds[i]) # coco 이미지 중에서 사람이 있는 이미지 아이디 전체를 불러옴
        reduced_person = sorted(person_ImgIds)[:person_img_num]
        imgIds += reduced_person
    else:
        imgIds += coco.getImgIds(catIds=catIds[i]) # category id가 사람이 아니면 그대로 이미지 아이디를 저장

imgIds = list(set(imgIds))
print(len(imgIds))
data_rootpath = "/home/minjae/mjseong/mmdetection/data/coco/"


#make img directory
try:
    if not(os.path.isdir(f'{data_rootpath}person{person_img_num}')):
        os.makedirs(os.path.join(f'{data_rootpath}person{person_img_num}'))
except:
    print('making directory error')

#extract image
total_img = os.listdir(f'{data_rootpath}coco_PersonCarTruck')
for imgId in imgIds:
    newid = str(imgId).zfill(12)+'.jpg'
    if newid in total_img:
        shutil.copy(f'{data_rootpath}coco_PersonCarTruck/{newid}',f'{data_rootpath}person{person_img_num}')

# add krri image
for krri_img in os.listdir(f"{data_rootpath}train_krri_data/images"):
    shutil.copy(f"{data_rootpath}train_krri_data/images/{krri_img}",f'{data_rootpath}person{person_img_num}')
print('finished copy images to newdirectory')

new_data={
    "images" :[
    ],
    "annotation" : [
    ],
    "categories" : [
        {'id': 1, 'name': 'person'},
        {'id': 2, 'name': 'car'},
        {'id': 3, 'name': 'truck'},
        {'id': 4, 'name': 'Fork_lift'},
        {'id': 5, 'name': 'Yard_Chassis'}
    ]
}

for imgId in imgIds:
    #image_information
    img_inform = coco.loadImgs(imgId)
    new_data["images"]+= img_inform
    #annotation_information
    gt_ann_ids = coco.getAnnIds(imgIds=imgId)
    gt_anns = coco.loadAnns(gt_ann_ids)
    new_gt_anns = []
    for i in range(len(gt_anns)):
        if gt_anns[i]['category_id'] == 1:
            gt_anns[i]['category_id'] = 1
            new_gt_anns.append(gt_anns[i])
        elif gt_anns[i]['category_id'] == 3:
            gt_anns[i]['category_id'] = 2
            new_gt_anns.append(gt_anns[i])
        elif gt_anns[i]['category_id'] == 8:
            gt_anns[i]['category_id'] = 3
            new_gt_anns.append(gt_anns[i])
        else:
            pass
    new_data['annotation'] += new_gt_anns


with open(f"{data_rootpath}annotations/person{person_img_num}.json","w") as json_file:
    json.dump(new_data,json_file)

print(f'person{person_img_num}.json')