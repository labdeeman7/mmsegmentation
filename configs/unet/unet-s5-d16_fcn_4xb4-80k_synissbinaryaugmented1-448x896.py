_base_ = [
    '../_base_/models/fcn_unet_s5-d16.py',
    '../_base_/datasets/syniss_binary_augmented_1.py', 
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]

crop_size = (448, 896)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor,
        decode_head=dict(num_classes=2),
        auxiliary_head=dict(num_classes=2),
        train_cfg=dict(),
        test_cfg=dict(mode='slide', crop_size=(256, 256), stride=(170, 170)))
        

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=50))

