# dataset settings
dataset_type = 'CocoDataset'
metainfo = {
    'classes' : ('chevrolet_malibu_sedan_2012_2016', 'chevrolet_malibu_sedan_2017_2019', 'chevrolet_spark_hatchback_2016_2021', 
           'chevrolet_trailblazer_suv_2021_', 'chevrolet_trax_suv_2017_2019','genesis_g80_sedan_2016_2020','genesis_g80_sedan_2021_', 
           'genesis_gv80_suv_2020_', 'hyundai_avante_sedan_2011_2015', 'hyundai_avante_sedan_2020_', 'hyundai_grandeur_sedan_2011_2016', 
           'hyundai_grandstarex_van_2018_2020', 'hyundai_ioniq_hatchback_2016_2019', 'hyundai_sonata_sedan_2004_2009', 
           'hyundai_sonata_sedan_2010_2014', 'hyundai_sonata_sedan_2019_2020', 'kia_carnival_van_2015_2020', 'kia_carnival_van_2021_',
           'kia_k5_sedan_2010_2015', 'kia_k5_sedan_2020_', 'kia_k7_sedan_2016_2020', 'kia_mohave_suv_2020_', 'kia_morning_hatchback_2004_2010',
           'kia_morning_hatchback_2011_2016', 'kia_ray_hatchback_2012_2017', 'kia_sorrento_suv_2015_2019', 'kia_sorrento_suv_2020_', 'kia_soul_suv_2014_2018',
           'kia_sportage_suv_2016_2020','kia_stonic_suv_2017_2019','renault_sm3_sedan_2015_2018','renault_xm3_suv_2020_','ssangyong_korando_suv_2019_2020',
           'ssangyong_tivoli_suv_2016_2020'),
    'palette': [
         (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
         (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
         (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
         (165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255),
         (0, 82, 0), (120, 166, 157), (110, 76, 0), (174, 57, 255),
         (199, 100, 0), (72, 0, 118), (255, 179, 240), (0, 125, 92),
         (209, 0, 151), (188, 208, 182), (0, 220, 176), (255, 99, 164),
         (92, 0, 73), (133, 129, 255), (78, 180, 255), (0, 228, 0),
    ]
}
data_root = '../data/'

# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection/coco/'

# Method 2: Use `backend_args`, `file_client_args` in versions before 3.0.0rc6
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))
backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
val_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        metainfo = metainfo,
        type=dataset_type,
        data_root=data_root,
        ann_file='K-fold_train1.json',
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        metainfo = metainfo,
        type=dataset_type,
        data_root=data_root,
        ann_file='K-fold_val1.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=val_pipeline,
        backend_args=backend_args))

val_evaluator = dict(
    type='CocoMetric',
    ann_file='K-fold_val1.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)

# inference on test dataset and
# format the output results for submission.
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img='test'),
        test_mode=True,
        pipeline=test_pipeline))
test_evaluator = dict(
    type='CocoMetric',
    metric='bbox',
    format_only=False,
    outfile_prefix='./work_dirs/run')
