_base_ = '../../_base_/default_runtime.py'

pretrained_model="path_to_clip-vit-base-patch16"
model = dict(
    type='CLIPSimilarity_split',
    visual_encoder=dict(type='VITCLIPPretrained_STAN',depth=4, return_mean=False, gradient_checkpointing=False, pretrained_model=pretrained_model),
    text_encoder=dict(type='CLIPTextPretrained',all_proj=True, pretrained_model=pretrained_model),
    multiview_dis=dict(type='MultiViewDis', dim=512, depth=8, heads=4, num_query_token=8),
    to_float32=True,
    frozen_layers=False,
    data_preprocessor=dict(
        type='MultiModalDataPreprocessor',
        preprocessors=dict(
            imgs=dict(
                type='ActionDataPreprocessor',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.3751],
                format_shape='NCHW'),
            text=dict(type='ActionDataPreprocessor', to_float32=False))),
    tau = 0.01,
    adapter=dict(type="DCM", input_dim = 512, tau = 100, communication=False),
    loss=dict(type='NormSoftmaxLoss'),
    multiview_loss = dict(type='MultiviewLoss'),
    ismultivew = True,
    concat=False,
    plus=True,
    concat_local=True,
    frozen=False)

load_from = "path_to_pretrained_mugstan_b16"


dataset_type = 'MsrvttDataset'
data_root = '../root/autodl-tmp/data/msrvtt_data'
file_client_args = dict(io_backend='disk')
train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='UniformSample', clip_len=12, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='CLIPTokenize', length=80),
    dict(type='PackActionInputs', collect_keys=('imgs', 'text'))
]
val_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='UniformSample', clip_len=12, num_clips=1, test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='CLIPTokenize', length=32),
    dict(type='PackActionInputs', collect_keys=('imgs', 'text'))
]
test_pipeline = val_pipeline

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file='train_9k.json',
        data_root=data_root,
        data_prefix=dict(video='videos'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file='test_JSFUSION.json',
        data_root=data_root,
        data_prefix=dict(video='videos'),
        pipeline=val_pipeline,
        test_mode=True))
test_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file='test_JSFUSION.json',
        data_root=data_root,
        data_prefix=dict(video='videos'),
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = dict(type='PostProc_RetrievalMetric', DSL=False)
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=100, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.05,
        by_epoch=True,
        begin=0,
        end=10,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=4.5,
        eta_min=0,
        by_epoch=True,
        begin=10,
        end=100,
        convert_to_iter_based=True)
]


optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=1e-06,
        betas=(0.9, 0.98),
        eps=1e-08,
        weight_decay=0.2),
    paramwise_cfg=dict(
        norm_decay_mult=0., bias_decay_mult=0.,
        bypass_duplicate=True,
        custom_keys={
            'STAN': dict(lr_mult=10.),
            'multiview_dis': dict(lr_mult=10.),
            'video_mlp': dict(lr_mult=10.),
    }),
    clip_grad=dict(max_norm=5, norm_type=2),
)


default_hooks = dict(checkpoint=dict(type='printBest_CheckpointHook', interval=-1, save_best='auto', rule='greater'))

auto_scale_lr = dict(enable=False, base_batch_size=128)