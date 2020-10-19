dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 4)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file = ['./coco_subset_images/annotations/filtered_train.json'],
        img_prefix=['./coco_subset_images/coco_carTruck'],
        # ann_file='data/open_images_downloader/image/annotations/carTruck_train-annotations-bbox.json',
        # img_prefix=data_root + 'train2017/',
        # img_prefix='data/open_images_downloader/image/train/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        # ann_file=[data_root + 'annotations/instances_val2017.json', data_root + 'annotations/car_truck_openImage_val.json']
        # separate_eval = False, # 연결한 데이터들 전체를 eval
        ann_file='./coco_subset_images/annotations/instances_val2017.json',
        img_prefix='./coco_subset_images/val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file='./coco_subset_images/annotations/instances_val2017.json',
        img_prefix='./coco_subset_images/val2017/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
