import copy
import os.path as osp
import os
import mmcv
import numpy as np
import json

from pycocotools.coco import COCO

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset
from mmcv import Config
from mmdet.apis import set_random_seed
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector


@DATASETS.register_module()
class KittiTinyDataset(CustomDataset):

    CLASSES = ('Car', 'Pedestrian', 'Cyclist')

    def load_annotations(self, ann_file):
        cat2label = {k: i for i, k in enumerate(self.CLASSES)}
        # load image list from file
        image_list = mmcv.list_from_file(self.ann_file) # train.txt에 있는 이미지 id를 가져옴
    
        data_infos = []
        # convert annotations to middle format
        for image_id in image_list:
            filename = f'{self.img_prefix}/{image_id}.jpeg'
            image = mmcv.imread(filename)
            height, width = image.shape[:2]

            # {'filename': ~~, 'width' : 1224, 'height' : 370}
            data_info = dict(filename=f'{image_id}.jpeg', width=width, height=height) 
    
            # load annotations
            label_prefix = self.img_prefix.replace('image_2', 'label_2')
            lines = mmcv.list_from_file(osp.join(label_prefix, f'{image_id}.txt')) # label.txt에 있는 파일의 첫 줄을 string으로 list에 담음
    
            content = [line.strip().split(' ') for line in lines]
            bbox_names = [x[0] for x in content]
            bboxes = [[float(info) for info in x[4:8]] for x in content]
    
            gt_bboxes = []
            gt_labels = []
            gt_bboxes_ignore = []
            gt_labels_ignore = []
    
            # filter 'DontCare'
            for bbox_name, bbox in zip(bbox_names, bboxes):
                if bbox_name in cat2label:
                    gt_labels.append(cat2label[bbox_name])
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
            

        return data_infos

@DATASETS.register_module()
class KrriDataset(CustomDataset):
    CLASSES = ('person', 'car', 'truck', 'fork_lift', 'yard_chassis')

    def load_annotations(self, ann_file):
        cat2label = {k: i for i, k in enumerate(self.CLASSES)}
        # load image list from file
        image_list = mmcv.list_from_file(self.ann_file) # train.txt에 있는 이미지 id를 가져옴
     
        coco_image_list = os.listdir("./data/coco/coco_PersonCarTruck/")
        data_infos = []
        # convert annotations to middle format
        for image_id in coco_image_list:
            filename = f'./data/coco/coco_PersonCarTruck/{image_id}'
            image = mmcv.imread(filename)
            height, width = image.shape[:2]

            # {'filename': ~~, 'width' : 1224, 'height' : 370}
            data_info = dict(filename=f'{image_id}', width=width, height=height) 
            
            # load annotations
            print("load_coco_annotations")
            with open("./data/coco/annotations/coco_PersonCarTruck_train.json", mode='r') as json_file:
                json_data = json.load(json_file)
                json_images = json_data["images"]
                json_annotations = json_data["annotations"]
                json_categories = json_data["categories"]

            print("make bbox")
            bbox_names = []
            bboxes = []
            
            for i in range(len(json_annotations)):
                import pdb; pdb.set_trace()
                if json_annotations[i]["image_id"] == int(data_info["filename"].rstrip(".jpg")):
                    bboxes.append(json_annotations[i]["bbox"])
                    if json_annotations[i]["category_id"] == json_categories[0]["id"]:
                        bbox_names.append(json_categories[0]["name"])
                    elif json_annotations[i]["category_id"] == json_categories[1]["id"]:
                        bbox_names.append(json_categories[1]["name"])
                    elif json_annotations[i]["category_id"] == json_categories[2]["id"]:
                        bbox_names.append(json_categories[2]["name"])
        
            gt_bboxes = []
            gt_labels = []
            gt_bboxes_ignore = []
            gt_labels_ignore = []
            # filter 'DontCare'
            for bbox_name, bbox in zip(bbox_names, bboxes):
                if bbox_name in cat2label:
                    gt_labels.append(cat2label[bbox_name])
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


        for image_id in image_list:
            filename = f'{self.img_prefix}/{image_id}.png'
            image = mmcv.imread(filename)
            height, width = image.shape[:2]

            # {'filename': ~~, 'width' : 1224, 'height' : 370}
            data_info = dict(filename=f'{image_id}.png', width=width, height=height) 
            
            # load annotations
            label_prefix = self.img_prefix.replace('images', 'labels')
            lines = mmcv.list_from_file(osp.join(label_prefix, f'{image_id}.txt')) # label.txt에 있는 파일의 첫 줄을 string으로 list에 담음
            content = [line.strip().split(' ') for line in lines]
            
            bbox_names = []
            bboxes = []
            for x in content :
                xmin = round((float(x[1]) - float(x[3])/2) * width, 2)
                ymin = round((float(x[1]) - float(x[3])/2) * height, 2)
                widths = round(float(x[3]) * width, 2)
                heights = round(float(x[4]) * height, 2)
                lists = [xmin, ymin, widths, heights]
                if x[0] == '5':
                    bbox_names.append('fork_lift')
                    bboxes.append(lists)
                elif x[0] == '6':
                    bbox_names.append('yard_chassis')
                    bboxes.append(lists)
                elif x[0] == '7' or x[0] == '8':
                    bbox_names.append('truck')
                    bboxes.append(lists)
                elif x[0] == '9':
                    bbox_names.append('car')
                    bboxes.append(lists)

            gt_bboxes = []
            gt_labels = []
            gt_bboxes_ignore = []
            gt_labels_ignore = []
            # filter 'DontCare'
            for bbox_name, bbox in zip(bbox_names, bboxes):
                if bbox_name in cat2label:
                    gt_labels.append(cat2label[bbox_name])
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
            

        return data_infos


cfg = Config.fromfile('./configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py')

# Modify dataset type and path
cfg.dataset_type = 'KrriDataset'
cfg.data_root = './data/krri/new_data/'

cfg.data.test.type = 'KittiTinyDataset'
cfg.data.test.data_root = './data/kitti_tiny/'
cfg.data.test.ann_file = 'train.txt'
cfg.data.test.img_prefix = 'training/image_2'

cfg.data.train.type = 'KrriDataset'
cfg.data.train.data_root = './data/krri/new_data/'
cfg.data.train.ann_file = 'train.txt'
cfg.data.train.img_prefix = 'images'

cfg.data.val.type = 'KittiTinyDataset'
cfg.data.val.data_root = './data/kitti_tiny/'
cfg.data.val.ann_file = 'val.txt'
cfg.data.val.img_prefix = 'training/image_2'

# modify num classes of the model in box head
# cfg.model.roi_head.bbox_head.num_classes = 3
# We can still use the pre-trained Mask RCNN model though we do not need to
# use the mask branch
cfg.load_from = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'

# Set up working dir to save files and logs.
cfg.work_dir = './tutorial_exps'

# The original learning rate (LR) is set for 8-GPU training.
# We divide it by 8 since we only use one GPU.
cfg.optimizer.lr = 0.0001
cfg.lr_config.warmup = None
cfg.log_config.interval = 50

# Change the evaluation metric since we use customized dataset.
cfg.evaluation.metric = 'mAP'
# We can set the evaluation interval to reduce the evaluation times
cfg.evaluation.interval = 12
# We can set the checkpoint saving interval to reduce the storage cost
cfg.checkpoint_config.interval = 1

# Set seed thus the results are more reproducible
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)


# We can initialize the logger for training and have a look
# at the final config used for training
print(f'Config:\n{cfg.pretty_text}')

# Build dataset
datasets = [build_dataset(cfg.data.train)]

# Build the detector
model = build_detector(
    cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
# Add an attribute for visualization convenience
model.CLASSES = datasets[0].CLASSES

# Create work_dir
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
train_detector(model, datasets, cfg, distributed=False, validate=True)