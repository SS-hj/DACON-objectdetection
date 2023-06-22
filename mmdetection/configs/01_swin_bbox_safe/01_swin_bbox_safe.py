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
                name = '01_swin_bbox_safe'
            ),
            )
    ])


runner = dict(type='EpochBasedRunner', max_epochs=25)

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


# customed swin-s
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth'  # noqa
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth'

model = dict(
    type='MaskRCNN',
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[96, 192, 384, 768]))


optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

lr_config = dict(warmup_iters=1000, step=[27, 33])
# runner = dict(max_epochs=36)

fp16 = dict(loss_scale=dict(init_scale=512))