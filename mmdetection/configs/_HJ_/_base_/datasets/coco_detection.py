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
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
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
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
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
        ann_file=data_root + 'K-fold_val1.json',
        img_prefix=data_root,
        classes=classes,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        img_prefix=data_root,
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox', save_best='bbox_mAP_75')
