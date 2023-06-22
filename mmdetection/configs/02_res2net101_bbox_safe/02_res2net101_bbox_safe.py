# Copyright (c) Shanghai AI Lab. All rights reserved.
# model 15. swin-s, origin+harder setting

_base_ = [
    # './cascade_mask_rcnn_r50_fpn.py',
    './cascade_rcnn_r50_fpn.py',
    './custom_dataset.py',
    './schedule_1x.py',
    '../_base_/default_runtime.py'
]


# yapf:disable
log_config = dict(
    interval= 100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook', interval=20,
             init_kwargs=dict(
                project='Dacon Object Detection',
                entity = 'jyjy230519',
                name = '2_res2net101_bbox_safe_same_seed'
            ),
            )
    ])


runner = dict(type='EpochBasedRunner', max_epochs=21)

workflow = [('train', 1)]

# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'

# save last 3 ckpts
checkpoint_config = dict(
    interval=1,
    max_keep_ckpts=3,
    save_last=True,
)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


# customed res2net 101

load_from = 'https://download.openmmlab.com/mmdetection/v2.0/res2net/cascade_rcnn_r2_101_fpn_20e_coco/cascade_rcnn_r2_101_fpn_20e_coco-f4b7b7db.pth'

model = dict(
    backbone=dict(
        type='Res2Net',
        depth=101,
        scales=4,
        base_width=26,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://res2net101_v1d_26w_4s')))


# runner = dict(max_epochs=36)

fp16 = dict(loss_scale=dict(init_scale=512))