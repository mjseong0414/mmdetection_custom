checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = "/home/minjae/mjseong/mmdetection/checkpoints/gfl_r101_fpn_mstrain_2x_coco_20200629_200126-dd12f847.pth"
resume_from = None
workflow = [('train', 1)]
