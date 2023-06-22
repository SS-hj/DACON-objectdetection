# dataset settings
dataset_type = 'CocoDataset'
data_root = '/opt/ml/DACON-objectdetection/data/'

classes = ('chevrolet_malibu_sedan_2012_2016', 'chevrolet_malibu_sedan_2017_2019', 'chevrolet_spark_hatchback_2016_2021', 
           'chevrolet_trailblazer_suv_2021_', 'chevrolet_trax_suv_2017_2019','genesis_g80_sedan_2016_2020','genesis_g80_sedan_2021_', 
           'genesis_gv80_suv_2020_', 'hyundai_avante_sedan_2011_2015', 'hyundai_avante_sedan_2020_', 'hyundai_grandeur_sedan_2011_2016', 
           'hyundai_grandstarex_van_2018_2020', 'hyundai_ioniq_hatchback_2016_2019', 'hyundai_sonata_sedan_2004_2009', 
           'hyundai_sonata_sedan_2010_2014', 'hyundai_sonata_sedan_2019_2020', 'kia_carnival_van_2015_2020', 'kia_carnival_van_2021_',
           'kia_k5_sedan_2010_2015', 'kia_k5_sedan_2020_', 'kia_k7_sedan_2016_2020', 'kia_mohave_suv_2020_', 'kia_morning_hatchback_2004_2010',
           'kia_morning_hatchback_2011_2016', 'kia_ray_hatchback_2012_2017', 'kia_sorrento_suv_2015_2019', 'kia_sorrento_suv_2020_', 'kia_soul_suv_2014_2018',
           'kia_sportage_suv_2016_2020','kia_stonic_suv_2017_2019','renault_sm3_sedan_2015_2018','renault_xm3_suv_2020_','ssangyong_korando_suv_2019_2020',
           'ssangyong_tivoli_suv_2016_2020')

# multi_scale = [(480, 1333), (680, 1333), (800, 1333)]

# simple
albu_train_transforms = [
    dict(type='BBoxSafeRandomCrop', erosion_rate=0.0, p=1.0),
    dict(type='GaussNoise', p=0.3),
    dict(type='MotionBlur', p=0.3),
    dict(type='RandomBrightnessContrast', p=0.4),
    dict(type='Blur', blur_limit=3, p=0.2),
    dict(type='CLAHE', p=0.3)
]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# augmentation strategy originates from DETR / Sparse RCNN
# added by soon

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(800, 1333), keep_ratio=True),
    dict(type='Pad', size_divisor=32),
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
           # 'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels'],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'img_norm_cfg',
                   'pad_shape', 'scale_factor'),
        )
]


test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MultiScaleFlipAug',
        img_scale=(800, 1333),
        # img_scale = multi_scale,
        flip=False,
        transforms=[
            dict(type='Resize'),
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
        pipeline=test_pipeline),
    test=dict(
        # _delete_=True,
        type=dataset_type,
        ann_file=data_root + 'test.json',
        img_prefix=data_root,
        classes=classes,
        # img_dir='test/',
        pipeline=test_pipeline))


evaluation = dict(interval=1, metric='bbox', save_best='bbox_mAP')
