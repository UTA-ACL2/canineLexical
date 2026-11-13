import os
from mmaction.apis import inference_recognizer, init_recognizer
from tqdm import tqdm

# ROOT = 'scene_pipeline'
ROOT = './'

config_path = os.path.join(ROOT, 'activity/configs/tsn_shiba_0525.py')
checkpoint_path = os.path.join(ROOT, 'activity/tsn0525.pth')


# build the model from a config file and a checkpoint file
model = init_recognizer(config_path, checkpoint_path, device="cuda")


def inference_activity(video_paths: list):
    """inference for videos"""
    results = []
    print('inferencing activity......')
    for path in tqdm(video_paths):
        res = inference_recognizer(model, path)
        results.append(res.pred_scores.item.cpu())
    return results


if __name__ == '__main__':
    demo = ['../demo/test.mp4']
    result = inference_activity(demo)
    