dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
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
dataset_A_train = dict(
    type = dataset_type,
    ann_file = data_root + 'annotations/filtered_CarTruck.json',
    img_prefix = data_root + 'coco_CarTruck/',
    pipeline = train_pipeline
)

dataset_B_train = dict(
    type = dataset_type,
    ann_file = './data/open_images_downloader/image/annotations/carTruck_train-annotations-bbox.json',
    img_prefix = './data/open_images_downloader/image/train/',
    pipeline = train_pipeline
)

dataset_A_val = dict(
    type = dataset_type,
    ann_file = data_root + 'annotations/filtered_CarTruck.json',
    img_prefix = data_root + 'coco_CarTruck/',
    pipeline = test_pipeline
)

dataset_B_val = dict(
    type = dataset_type,
    ann_file = './data/open_images_downloader/image/annotations/carTruck_train-annotations-bbox.json',
    img_prefix = './data/open_images_downloader/image/train/',
    pipeline = test_pipeline
)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    # train=dict(
    #     type='ClassBalancedDataset',
    #     oversample_thr=0.3,
    #     dataset=dict(
    #         type=dataset_type,
    #         ann_file=data_root + 'annotations/filtered_train.json',
    #         img_prefix=data_root + 'coco_PersonCarTruckCatDog_train/',
    #         pipeline=train_pipeline,
    #         )),
    train=[
        dataset_A_train,
        dataset_B_train
    ],
    val=dict(
        type='ConcatDataset',
        datasets = [dataset_A_val, dataset_B_val],
        # separate_eval = False, # 연결한 데이터들 전체를 eval
        ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
