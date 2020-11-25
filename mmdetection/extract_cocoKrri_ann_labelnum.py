from pycocotools.coco import COCO
import requests, os
import json
import shutil
'''
coco image 폴더에서 사람이 속한 이미지를 
coco annotation 파일과 
'''
coco = COCO(os.path.abspath('./data/coco/annotations/instances_train2017.json'))
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

old_cat_Id = coco.getCatIds()
new_cat_Id = [1,2,3,4,5]

#initialize
catIds = []
train_imgIds = []
val_imgIds = []
person_img_num = 10000 #########
percentage = 20
want_class = ['person','car','truck']

for classes in want_class:
    catIds += coco.getCatIds(catNms=[classes])
print(catIds)

for i in range(len(want_class)):
    if catIds[i] == 1:
        person_ImgIds = coco.getImgIds(catIds=catIds[i])
        reduced_person_train = sorted(person_ImgIds)[:person_img_num]
        reduced_person_val = sorted(person_ImgIds)[person_img_num:person_img_num + int(person_img_num/2)]
        train_imgIds += reduced_person_train
        val_imgIds += reduced_person_val
    # else:
    #     train_imgIds += coco.getImgIds(catIds=catIds[i])
        

train_imgIds = list(set(train_imgIds))
val_imgIds = list(set(val_imgIds))

data_rootpath = "/home/minjae/mjseong/mmdetection/data/coco"
krri_rootpath = os.path.abspath('./data/coco/train_krri_data')

#make img directory
try:
    if not(os.path.isdir(f'{data_rootpath}/coco_krri/person{person_img_num}_train{percentage}/')): 
        os.makedirs(os.path.join(f'{data_rootpath}/coco_krri/person{person_img_num}_train{percentage}/')) 
        os.makedirs(os.path.join(f'{data_rootpath}/coco_krri/person{person_img_num}_train{percentage}/images')) 
        os.makedirs(os.path.join(f'{data_rootpath}/coco_krri/person{person_img_num}_train{percentage}/labels'))

    if not(os.path.isdir(f'{data_rootpath}/coco_krri/person{person_img_num}_val{percentage}/')): 
        os.makedirs(os.path.join(f'{data_rootpath}/coco_krri/person{person_img_num}_val{percentage}/')) 
        os.makedirs(os.path.join(f'{data_rootpath}/coco_krri/person{person_img_num}_val{percentage}/images')) # coco + krri images
        os.makedirs(os.path.join(f'{data_rootpath}/coco_krri/person{person_img_num}_val{percentage}/labels'))
except:
    print('making directory error')

#extract coco train images
total_img = os.listdir(f'{data_rootpath}/coco_PersonCarTruck')
for imgId in train_imgIds:
    newid = str(imgId).zfill(12)+'.jpg'
    if newid in total_img:
        shutil.copy(f'{data_rootpath}/coco_PersonCarTruck/{newid}',f'{data_rootpath}/coco_krri/person{person_img_num}_train{percentage}/images') 

#extract coco val images
for imgId in val_imgIds:
    newid = str(imgId).zfill(12)+'.jpg'
    if newid in total_img:
        shutil.copy(f'{data_rootpath}/coco_PersonCarTruck/{newid}',f'{data_rootpath}/coco_krri/person{person_img_num}_val{percentage}/images')
        
total_krri_images = os.listdir(krri_rootpath+"/images/")
val_images_num = int(len(total_krri_images) * (percentage/100)) # val image 갯수

train_krri_images = sorted(total_krri_images)[val_images_num:]
val_krri_images = sorted(total_krri_images)[:val_images_num]

#extract krri train images
for train_imgId in train_krri_images:
    train_label = train_imgId.split(".")[0]+".txt"
    shutil.copy(f'{krri_rootpath}/images/{train_imgId}',f'{data_rootpath}/coco_krri/person{person_img_num}_train{percentage}/images')
    shutil.copy(f'{krri_rootpath}/labels/{train_label}',f'{data_rootpath}/coco_krri/person{person_img_num}_train{percentage}/labels')

#extract krri val images
for val_imgId in val_krri_images:
    val_label = val_imgId.split(".")[0]+".txt"
    shutil.copy(f'{krri_rootpath}/images/{val_imgId}',f'{data_rootpath}/coco_krri/person{person_img_num}_val{percentage}/images')
    shutil.copy(f'{krri_rootpath}/labels/{val_label}',f'{data_rootpath}/coco_krri/person{person_img_num}_val{percentage}/labels')

print('finished copy images to newdirectory')

new_data={
    "images" :[
    ],
    "annotations" : [
    ],
    "categories" : [
        {'id': 1, 'name': 'person'},
        {'id': 2, 'name': 'car'},
        {'id': 3, 'name': 'truck'},
        {'id': 4, 'name': 'Fork_lift'},
        {'id': 5, 'name': 'Yard_Chassis'}
    ]
}

try:
    for imgId in train_imgIds:
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
        new_data['annotations'] += new_gt_anns
finally:
    with open(f"{data_rootpath}/annotations/coco_krri/person{person_img_num}_train_{percentage}.json","w") as json_file:
        json.dump(new_data,json_file)

try:
    for imgId in val_imgIds:
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
        new_data['annotations'] += new_gt_anns
finally:
    with open(f"{data_rootpath}/annotations/coco_krri/person{person_img_num}_val_{percentage}.json","w") as json_file:
        json.dump(new_data,json_file)

print(f'finish')