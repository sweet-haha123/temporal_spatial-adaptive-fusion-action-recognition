_base_ = ['../../_base_/default_runtime.py']

# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(
        pretrained=  # noqa: E251
                '../weight/vit-base-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-860a3cd3.pth',  # noqa: E501
        type='PretrainVisionTransformer',
        img_size=224,
        patch_size=16,
        encoder_in_chans=3,
        encoder_num_classes=0,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
                            ),
    cls_head=dict(
        type='TimeSformerHead',
        num_classes=12,
        in_channels=1536,
        average_clips='prob'),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCTHW'))

# # dataset settings
# dataset_type = 'VideoDataset'
# data_root_val = 'data/kinetics400/videos_val'
# ann_file_test = 'data/kinetics400/kinetics400_val_list_videos.txt'
#
# test_pipeline = [
#     dict(type='DecordInit'),
#     dict(
#         type='SampleFrames',
#         clip_len=16,
#         frame_interval=4,
#         num_clips=5,
#         test_mode=True),
#     dict(type='DecordDecode'),
#     dict(type='Resize', scale=(-1, 224)),
#     dict(type='ThreeCrop', crop_size=224),
#     dict(type='FormatShape', input_format='NCTHW'),
#     dict(type='PackActionInputs')
# ]
#
# test_dataloader = dict(
#     batch_size=1,
#     num_workers=8,
#     persistent_workers=True,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         ann_file=ann_file_test,
#         data_prefix=dict(video=data_root_val),
#         pipeline=test_pipeline,
#         test_mode=True))
#
# test_evaluator = dict(type='AccMetric')
# test_cfg = dict(type='TestLoop')

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
    dict(type='SampleFrames', clip_len=8, frame_interval=32, num_clips=1),
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
        frame_interval=32,
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
    type='EpochBasedTrainLoop',max_epochs=150, val_begin=1, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    type='AmpOptimWrapper', #add by qyh, automatic mixed training
    optimizer=dict(
        # type='SGD', lr=0.006, momentum=0.8, weight_decay=1e-4, nesterov=True),
        type='SGD', lr=0.005, momentum=0.9, weight_decay=1e-4, nesterov=True),
    paramwise_cfg=dict(
        custom_keys={
            '.backbone.cls_token': dict(decay_mult=0.0),
            '.backbone.pos_embed': dict(decay_mult=0.0),
            '.backbone.time_embed': dict(decay_mult=0.0)
        }),
    clip_grad=dict(max_norm=40, norm_type=2))

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=150,
        by_epoch=True,
        # milestones=[5, 140],
        milestones=[50, 100],
        gamma=0.1)
]

default_hooks = dict(checkpoint=dict(type='CheckpointHook',interval=5,max_keep_ckpts=1,save_best='auto'),
                     logger=dict(type='LoggerHook',interval=5))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (8 samples per GPU).
auto_scale_lr = dict(enable=True, base_batch_size=64)
load_from='../weight/vit-base-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-860a3cd3.pth'
# resume=True
# resume=True

