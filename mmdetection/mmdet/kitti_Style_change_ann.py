import copy
import os.path as osp
import os

import mmcv
import numpy as np
import json

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset
def main():
    # CLASSES = ('Car', 'Pedestrian', 'Cyclist')
    ClassId = ('5', '6', '7', '8', '9')
    img_prefix = "./data/krri/new_data/images"
    cat2label = {k: i for i, k in enumerate(ClassId)}
    # load image list from file
    image_list = mmcv.list_from_file("./data/krri/new_data/train.txt")

    data_infos = []
    # convert annotations to middle format
    for image_id in image_list:
        filename = f'{img_prefix}/{image_id}.png'
        image = mmcv.imread(filename)
        height, width = image.shape[:2]

        data_info = dict(filename=f'{image_id}.png', width=width, height=height)

        # load annotations
        label_prefix = img_prefix.replace('images', 'labels')
        lines = mmcv.list_from_file(osp.join(label_prefix, f'{image_id}.txt'))

        content = [line.strip().split(' ') for line in lines]
        # bbox_names = [x[0] for x in content]
        bbox_id = [x[0] for x in content] # category id
        # bbox = []
        # bboxes = []
        # for x in content:
        #     Xmin = float(x[1]) - float(x[3])/2
        #     Ymin = float(x[2]) - float(x[4])/2
        #     Width = float(x[3])
        #     Height = float(x[4])
        #     bbox.append(Xmin)
        #     bbox.append(Ymin)
        #     bbox.append(Width)
        #     bbox.append(Height)
        #     bboxes.append(bbox)
        
        #     del bbox[:]
        bboxes = [[float(info) for info in x[1:]] for x in content]

        i = 0
        while True:
            bboxes[i][0] = (bboxes[i][0] - bboxes[i][2]/2) * data_info["width"]
            bboxes[i][1] = (bboxes[i][1] - bboxes[i][3]/2) * data_info["height"]
            i += 1
            if i == len(bboxes) - 1:
                break
        
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_labels_ignore = []

        # filter 'DontCare'
        for bbox_id, bbox in zip(bbox_id, bboxes):
            if bbox_id in cat2label:
                gt_labels.append(cat2label[bbox_id])
                gt_bboxes.append(bbox)
            else:
                gt_labels_ignore.append(-1)
                gt_bboxes_ignore.append(bbox)

        data_anno = dict(
            bboxes=np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4),
            labels=np.array(gt_labels, dtype=np.long),
            bboxes_ignore=np.array(gt_bboxes_ignore,
                                    dtype=np.float32).reshape(-1, 4),
            labels_ignore=np.array(gt_labels_ignore, dtype=np.long))

        data_info.update(ann=data_anno)
        data_infos.append(data_info)
        import pdb; pdb.set_trace()

    with open("./data/krri/new_data/krri_annotation.json", mode='w') as f:
        json.dump(data_infos, f)
        

if __name__ == "__main__":
    main()