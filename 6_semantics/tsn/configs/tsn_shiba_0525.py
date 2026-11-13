# _base_ = [
#     '/home/chunhao/AnimalLanguage/scene_pipeline/activity/configs/tsn_r50.py', 
#     '/home/chunhao/AnimalLanguage/scene_pipeline/activity/configs/sgd_100e.py',
#     '/home/chunhao/AnimalLanguage/scene_pipeline/activity/configs/default_runtime.py'
# ]


_base_ = [
    './tsn_r50.py', 
    './sgd_100e.py',
    './default_runtime.py'
]
file_client_args = dict(io_backend='disk')

test_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=25,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]

val_evaluator = [dict(type='AccMetric')]
test_evaluator = val_evaluator

model = dict(
    cls_head=dict(num_classes=16))
