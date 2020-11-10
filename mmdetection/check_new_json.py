from pycocotools.coco import COCO
import requests, os
import json
import shutil
import cv2

new_json = 'person10000_krri_v2.json'
person_img_num = 10000

with open(f"./coco/annotations/{new_json}","r") as json_file:
    json_data = json.load(json_file)
    annotations = json_data['annotations']
    images = json_data["images"]
    category_id = set()
    image_id = set()
    image_id2 = set()
    
    annotation_id = set()
    for i in range(len(annotations)):
        category_id.add(annotations[i]['category_id'])
        image_id.add(annotations[i]['image_id'])
        annotation_id.add(annotations[i]['id'])
    if len(category_id) != len(json_data['categories']):
        print("categories number error")
    elif len(annotation_id) != len(json_data['annotations']):
        print("annotations number error")
    elif len(image_id) != len(json_data['images']):
        print("annotations number error")
    
    image_list = list(image_id)

    
    for i in range(len(images)):
        image_id2.add(images[i]['id'])
    if len(image_id2) != len(json_data['images']):
        print("image number error")
    
    import pdb ; pdb.set_trace()


    remove_img = list(image_id2-image_id)
    
# remove_img = [100001280, 100001281, 100001282, 100001283, 100000772, 100000773, 100000774, 100000775, 100001284, 100001285, 100001288, 100000779, 100000780, 100001289, 100001287, 100000785, 100000786, 100000787, 100000788, 100000894, 100000895, 100000896, 100000951, 100000442, 100000444, 100000445, 100000463, 100001045, 100001046, 100001047, 100001048, 100001057, 100000036, 100001060, 100001073, 100001109, 100000614, 100000617, 100000618, 100000623, 100000624, 100000625, 100000631, 100000632, 100000633, 100000634, 100000635, 100000638, 100000639, 100000640, 100000651, 100000653, 100000676, 100001190, 100001191, 100001192, 100000682, 100000683, 100001195, 100001196, 100001198, 100001199, 100001200, 100001201, 100001202, 100001204, 100001205, 100001206, 100001207, 100001208, 100001209, 100001210, 100001211, 100001212, 100001213, 100001215, 100001216, 100001220, 100001225, 100001226, 100001227, 100001228, 100001229, 100001230, 100001231, 100001232, 100001233, 100001234, 100001235, 100001236, 100001237, 100001238, 100001239, 100001240, 100001241, 100001242, 100001243, 100001244, 100001245, 100001246, 100001247, 100001248, 100001249, 100001250, 100001252, 100001253, 100001254, 100001255, 100001256, 100001257, 100001258, 100001259, 100001261, 100001262, 100001263, 100001264, 100001265, 100001266, 100001267, 100001269, 100001270, 100001271, 100001272, 100001273, 100001274, 100001275, 100001276, 100001277, 100001278, 100001279]
# #check image and find if this code has error
# print(len(remove_img))
# data_rootpath = "/home/minjae/mjseong/mmdetection/data/krri/new_data/"  
# image_path = f'{data_rootpath}labels/'
# print(len(os.listdir(image_path)))
# for img in remove_img:
#     image_name =str(img).zfill(12)+'.txt'
#     os.remove(f'{image_path}{image_name}')

# if len(os.listdir(image_path)) == int(1490-len(remove_img)):
#     print('good job')
