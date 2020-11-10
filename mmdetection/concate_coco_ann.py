from pycocotools.coco import COCO
import requests, os
import json
import shutil
import cv2

old_json = 'person10000.json'
new_json = 'person10000_krri_v2.json'
new_cat_Id = [1,2,3,4,5]

new_data={
    "images" :[
    ],
    "annotations" : [
    ],
    "categories" : [
    ]
}

#we use label 5=Fork_lift, 6=Yard_Chassis, 7,8=truck, 9=cars

data_rootpath = "/home/minjae/mjseong/mmdetection/data/coco/train_krri_data"
old_label_path = f'{data_rootpath}/labels'
image_path = f'{data_rootpath}/images'
image_dict = []
annotation_dict = []
instance_index = 0
for label in os.listdir(old_label_path):
    image_id = label.split('.')[0]
    image_Id = int(image_id.lstrip('0'))
    file_name = f'{image_id}.png'
    img = cv2.imread(f'{image_path}/{file_name}')
    img_height = img.shape[0]
    img_width = img.shape[1]
    image_inform = {'file_name':file_name, 'height':img_height, 'width':img_width,'id':image_Id}
    image_dict.append(image_inform)
    with open(f'{old_label_path}/{label}','r') as label_txt:
        lines =  label_txt.readlines()
        for i in range(len(lines)):
            new_line = lines[i].rstrip('\n').split()
            if new_line[0] in ['5','6','7','8','9']:
                xmin = round(img_width*(float(new_line[1])-float(new_line[3])/2),2)
                ymin = round(img_height*(float(new_line[2])-float(new_line[4])/2),2)
                bbox_width = round(float(new_line[3])*img_width,2)
                bbox_height = round(float(new_line[4])*img_height,2)
                bbox = [xmin,ymin,bbox_width,bbox_height]
                instance_index += 1
                instance_id = 2020418600000+instance_index
                #define category
                if new_line[0] =='5':
                    category_id = 4
                elif new_line[0] == '6':
                    category_id = 5
                elif new_line[0] == '7' or new_line[0] == '8':
                    category_id = 3
                elif new_line[0] == '9':
                    category_id = 2
                annotations = {'image_id':image_Id,'bbox':bbox,'category_id':category_id,'id':instance_id,'iscrowd':0,'area':200}
                annotation_dict.append(annotations)


with open(f"./data/coco/annotations/{old_json}","r") as json_file:
    json_data = json.load(json_file)
    json_data["images"] += image_dict
    json_data["annotations"] += annotation_dict

with open(f"./data/coco/annotations/{new_json}","w") as json_file2:
    json.dump(json_data,json_file2)
