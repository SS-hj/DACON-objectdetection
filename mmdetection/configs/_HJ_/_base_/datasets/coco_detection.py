# dataset settings
dataset_type = 'CocoDataset'
data_root = '../data/'

classes = ('chevrolet_malibu_sedan_2012_2016', 'chevrolet_malibu_sedan_2017_2019', 'chevrolet_spark_hatchback_2016_2021', 
           'chevrolet_trailblazer_suv_2021_', 'chevrolet_trax_suv_2017_2019','genesis_g80_sedan_2016_2020','genesis_g80_sedan_2021_', 
           'genesis_gv80_suv_2020_', 'hyundai_avante_sedan_2011_2015', 'hyundai_avante_sedan_2020_', 'hyundai_grandeur_sedan_2011_2016', 
           'hyundai_grandstarex_van_2018_2020', 'hyundai_ioniq_hatchback_2016_2019', 'hyundai_sonata_sedan_2004_2009', 
           'hyundai_sonata_sedan_2010_2014', 'hyundai_sonata_sedan_2019_2020', 'kia_carnival_van_2015_2020', 'kia_carnival_van_2021_',
           'kia_k5_sedan_2010_2015', 'kia_k5_sedan_2020_', 'kia_k7_sedan_2016_2020', 'kia_mohave_suv_2020_', 'kia_morning_hatchback_2004_2010',
           'kia_morning_hatchback_2011_2016', 'kia_ray_hatchback_2012_2017', 'kia_sorrento_suv_2015_2019', 'kia_sorrento_suv_2020_', 'kia_soul_suv_2014_2018',
           'kia_sportage_suv_2016_2020','kia_stonic_suv_2017_2019','renault_sm3_sedan_2015_2018','renault_xm3_suv_2020_','ssangyong_korando_suv_2019_2020',
           'ssangyong_tivoli_suv_2016_2020')

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

multi_scale = [(800, 1333),(1600, 2666)]

albu_train_transforms = [
        dict(type='RandomCrop', height=500, width=300, p=0.3),
		dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MotionBlur', blur_limit=11, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0),
        ],
        p=0.5),
		dict(
        type='OneOf',
        transforms=[
            dict(type='ISONoise', p=1.0),
            dict(type='RandomGamma', p=1.0),
            dict(type='HueSaturationValue', p=1.0),
            dict(type='ChannelShuffle', p=1.0),
            dict(type='RGBShift', p=1.0),
            dict(type='CLAHE', clip_limit=(1, 10), p=1.0),
            dict(type='RandomBrightnessContrast', p=1.0), 
        ],
        p=0.6),
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=multi_scale, multiscale_mode='range', keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
    type='MultiScaleFlipAug',
        img_scale=multi_scale,
        flip=False,
        transforms=[
            dict(type='Resize', multiscale_mode='range', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=multi_scale,
        flip=False,
        transforms=[
            dict(type='Resize', multiscale_mode='range', keep_ratio=True),
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
        ann_file=data_root + 'K-fold_train1.json',
        img_prefix=data_root,
        classes=classes,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'augmented_val.json',
        img_prefix=data_root,
        classes=classes,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/test.json',
        img_prefix=data_root,
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox', save_best='bbox_mAP')
