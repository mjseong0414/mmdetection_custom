# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001) # 원래 lr은 0.02임. 근데 이건 8개의 GPU사용. 2(img/gpu)일때임. 우린 1개 GPU를 쓰기 때문에 lr = 0.0025로 설정해줘야함
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[2, 4, 6]) #에폭 시작할 때 lr을 1/10 줄임
total_epochs = 12