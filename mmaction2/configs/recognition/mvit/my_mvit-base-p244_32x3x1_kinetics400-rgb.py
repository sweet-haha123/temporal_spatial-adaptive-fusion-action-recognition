_base_ = [
    '../../_base_/models/mvit_small.py', '../../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        arch='base',
        temporal_size=32,
        drop_path_rate=0.3,
    ),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[114.75, 114.75, 114.75],
        std=[57.375, 57.375, 57.375],
        blending=dict(
            type='RandomBatchAugment',
            augments=[
                dict(type='MixupBlending', alpha=0.8, num_classes=400),
                dict(type='CutmixBlending', alpha=1, num_classes=400)
            ]),
        format_shape='NCTHW'),
)

# dataset settings
dataset_type = 'RawframeDataset'
data_root = '../data/vis_data/rawframes'
data_root_val = '../data/vis_data/rawframes'
split = 1
ann_file_train = f'../data/vis_data/vis_data_train_split_{split}_rawframes.txt'
ann_file_val = f'../data/vis_data/vis_data_val_split_{split}_rawframes.txt'
ann_file_test = f'../data/vis_data/vis_data_val_split_{split}_rawframes.txt'

ir_file_train = '../data/ir_data/ir_data_train_split_1_rawframes.txt'
ir_file_val = '../data/ir_data/ir_data_val_split_1_rawframes.txt'
ir_file_test = '../data/ir_data/ir_data_val_split_1_rawframes.txt'

file_client_args = dict(io_backend='disk')

train_pipeline = [
    dict(type='SampleFrames', clip_len=8, frame_interval=8, num_clips=1),
    dict(type='RawFrameDecode', **file_client_args),
    dict(type='RandomRescale', scale_range=(256, 320)),
    dict(type='RandomCrop', size=224),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=8,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode', **file_client_args),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
test_pipeline = val_pipeline
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        ir_file=ir_file_train,
        data_prefix=dict(img=data_root),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        ir_file=ir_file_val,
        data_prefix=dict(img=data_root_val),
        pipeline=val_pipeline,
        test_mode=True))
test_dataloader =val_dataloader

val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator
visualizer=dict(type='Visualizer', vis_backends=[dict(type='TensorboardVisBackend')])

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=200, val_begin=1, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

base_lr = 1.6e-3
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW', lr=base_lr, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0),
    clip_grad=dict(max_norm=1, norm_type=2))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.01,
        by_epoch=True,
        begin=0,
        end=30,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=200,
        eta_min=base_lr / 100,
        by_epoch=True,
        begin=30,
        end=200,
        convert_to_iter_based=True)
]

default_hooks = dict(
    checkpoint=dict(interval=3, max_keep_ckpts=5), logger=dict(interval=100))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (8 samples per GPU).
auto_scale_lr = dict(enable=True, base_batch_size=512 // 2)
