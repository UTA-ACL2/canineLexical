import argparse
import bisect
import json
import logging
import os
import shutil
import cv2
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor

import pandas as pd

import ffmpeg
from datetime import datetime

import numpy as np
import torch
from mmaction.apis import inference_recognizer, init_recognizer
from moviepy.video.io.VideoFileClip import VideoFileClip
from tqdm import tqdm
from TransnetV2 import TransNetV2
from adtk.data import validate_series
from adtk.detector import PersistAD

from util import set_seed, log_init

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(ROOT, 'activity/configs/tsn_shiba_0525.py')
checkpoint_path = os.path.join(ROOT, 'activity/tsn0525.pth')
cate_p = os.path.join(ROOT, 'activity/shiba_activity_categories_en.txt')

"""
conda activate context_ch
cd /home2/xingyuan/animal_language/v2/code/context/mmaction

python work/activity.py \
       --input_path /home2/xingyuan/animal_language/v2/data/context/input_data/train_split/train_2704.json \
       --log_root /home2/xingyuan/animal_language/v2/log \
       --device 3

python work/activity.py \
       --input_path /home2/xingyuan/animal_language/v2/data/context/input_data/train_split/train_27040.json \
       --log_root /home2/xingyuan/animal_language/v2/log \
       --device 3

python work/activity.py \
       --input_path /home2/xingyuan/animal_language/v2/data/context/input_data/train_split/train_28392.json \
       --log_root /home2/xingyuan/animal_language/v2/log \
       --device 3

python work/activity.py \
       --input_path /home2/xingyuan/animal_language/v2/data/context/input_data/train_split/train_29744.json \
       --log_root /home2/xingyuan/animal_language/v2/log \
       --device 2

python work/activity.py \
       --input_path /home2/xingyuan/animal_language/v2/data/context/input_data/train_split/train_31096.json \
       --log_root /home2/xingyuan/animal_language/v2/log \
       --device 2

python work/activity.py \
       --input_path /home2/xingyuan/animal_language/v2/data/context/input_data/train_split/train_32448.json \
       --log_root /home2/xingyuan/animal_language/v2/log \
       --device 2

python work/activity.py \
       --input_path /home2/xingyuan/animal_language/v2/data/context/input_data/train_split/train_4056.json \
       --log_root /home2/xingyuan/animal_language/v2/log \
       --device 1

python work/activity.py \
       --input_path /home2/xingyuan/animal_language/v2/data/context/input_data/train_split/train_5408.json \
       --log_root /home2/xingyuan/animal_language/v2/log \
       --device 1

python work/activity.py \
       --input_path /home2/xingyuan/animal_language/v2/data/context/input_data/train_split/train_6760.json \
       --log_root /home2/xingyuan/animal_language/v2/log \
       --device 1

python work/activity.py \
       --input_path /home2/xingyuan/animal_language/v2/data/context/input_data/train_split/train_8112.json \
       --log_root /home2/xingyuan/animal_language/v2/log \
       --device 0

python work/activity.py \
       --input_path /home2/xingyuan/animal_language/v2/data/context/input_data/train_split/train_9464.json \
       --log_root /home2/xingyuan/animal_language/v2/log \
       --device 0

"""

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True,
                        help='The path of input (JSON FILE)')
    parser.add_argument('--output_path', type=str, required=False,
                        help='The path of output (JSONLINES FILE)')
    parser.add_argument('--output_video_tmp_root', type=str, required=False,
                        help='The root of video')
    parser.add_argument('--log_root', type=str, required=True,
                        help='The root of log')

    parser.add_argument('--device', type=int, required=False,
                        default=0,
                        help='The number of GPU (-1 is CPU)')
    parser.add_argument('--offset', type=float, required=False,
                        default=5.,
                        help='The offset')
    parser.add_argument('--clip_length', type=float, required=False,
                        default=2.,
                        help='The clip length')
    parser.add_argument('--seed', type=int, required=False,
                        default=42,
                        help='The seed')
    return parser.parse_args()

##########
def find_contiguous_regions(activity_array):
    """Find contiguous regions from bool valued numpy.array.
    left close and right open
    """
    # Find the changes in the activity_array
    change_indices = np.logical_xor(activity_array[1:],
                                    activity_array[:-1]).nonzero()[0]
    # Shift change_index with one, focus on frame after the change.
    change_indices += 1

    if activity_array[0]:
        # If the first element of activity_array is True add 0 at the beginning
        change_indices = np.r_[0, change_indices]
    if activity_array[-1]:
        # If the last element of activity_array is True, add the length of the array
        change_indices = np.r_[change_indices, activity_array.size]

    # Reshape the result into two columns
    return change_indices.reshape((-1, 2))


def connect(pairs, n=60):
    """Connects two adjacent clusters if their distance is <= n

    :param pairs: Clusters of iterateables e.g., [(1,5),(7,10)]
    :param n: distance between two clusters
    """
    if len(pairs) == 0:
        return []
    start_, end_ = pairs[0]
    new_pairs = []
    for i, (next_item, cur_item) in enumerate(zip(pairs[1:], pairs[0:])):
        end_ = next_item[1]
        if next_item[0] - cur_item[1] <= n:
            pass
        else:
            new_pairs.append((start_, cur_item[1]))
            start_ = next_item[0]
    new_pairs.append((start_, end_))
    return new_pairs


def persist_ad(array, window=30, th=100):
    """persist anomaly detection

    Parameters
    ----------
    array: numpy.ndarray, 1-d time series data
    window: the width of sliding window for statistic
    th: threshold for the anomaly detection
    """
    # add time index for the adtk
    date = pd.date_range(start='2023-4-4', periods=array.size, freq='S')
    df = pd.DataFrame({'time': date, 'data': array}).set_index('time')
    df = validate_series(df)

    # anomaly detection
    persist_ad = PersistAD(window=window, c=th, side='positive', agg='median', min_periods=None)
    anomalies = persist_ad.fit_detect(df)
    anomalies[anomalies.isna()] = False

    return np.array(anomalies, dtype=bool).reshape(-1, )


def segment_one(cut_res: dict):
    """based on the devide points and anomaly region, get the segment result of a video, and the stable regions in the segments"""
    left, right, stable_left, stable_right, sample_frames = [], [], [], [], []
    stable_l = 0
    l = 0
    for i_devide in range(cut_res['devide_points'].size):
        stable_r = cut_res['anomaly_regions'][i_devide][0]
        r = cut_res['devide_points'][i_devide]

        left.append(l)
        stable_left.append(stable_l)
        stable_right.append(stable_r)
        right.append(r)
        sample_frames.append(cut_res['sample_frames'][i_devide])

        l = r
        stable_l = cut_res['anomaly_regions'][i_devide][1]
    left.append(l)
    stable_left.append(stable_l)
    stable_right.append(cut_res['stable_frames'].size)
    right.append(cut_res['stable_frames'].size)
    sample_frames.append(cut_res['sample_frames'][-1])
    return {'semgent_id': range(len(left)),
                               'left': left,
                               'right': right,
                               'stable_left': stable_left,
                               'stable_right': stable_right,
                               'sample_frames': sample_frames}


def cut_one_video(pred_arr):
    """process on a video, return the anomaly regions and devided points
    1. transition detection by TransnetV2
    2. post-process of TransnetV2 predicts, by adtk, get the anomaly regions
    3. union the regions which gap by less that 60 frames
    4. find the most left and anomaly frame in anomaly regions, as the devide points
    5. after deviding, the segments with no anomaly points are stable regions, which can be used for location classification and activity classification

    Parameters
    ----------
    filepath: Transnet prediction
    th: the threshold for selecting stable frames
    """
    # # arange the TransNet preds into a 1-d array
    # with open(filepath, 'r') as fp:
    #     trans_pred = fp.readlines()
    #     pred_arr = np.array([float(d.strip().split(' ')[0]) for d in trans_pred])

    # persist anomaly detect
    anomaly_arr = persist_ad(pred_arr)
    anomaly_arr[np.isnan(anomaly_arr)] = 0

    # connect anomaly points into continue regions
    anomaly_regions = find_contiguous_regions(anomaly_arr)
    anomaly_regions = connect(anomaly_regions, n=60)

    # find stable regions for scene classification
    stable_frames = np.ones_like(pred_arr, dtype=int)
    for i_region in range(len(anomaly_regions)):
        (left, right) = anomaly_regions[i_region]
        stable_frames[left: right] = 0

    # remove the devide_points that are too close to endpoints of the video
    _anomaly_regions = []
    for i_region in range(len(anomaly_regions)):
        (left, right) = anomaly_regions[i_region]
        if (left < 60 or right > pred_arr.size - 60):
            continue
        _anomaly_regions.append(anomaly_regions[i_region])
    anomaly_regions = _anomaly_regions

    # locate the exact frame that devide the clips by transition
    # use the largest, leftmost point in the anomaly region
    devide_points = []
    for i_region in range(len(anomaly_regions)):
        (left, right) = anomaly_regions[i_region]
        devide_idx = np.argmax(pred_arr[left: right])
        devide_points.append(devide_idx + left)

    sample_frames = []
    for i_trans in range(len(devide_points) + 1):
        if (i_trans > 0):
            left = devide_points[i_trans - 1]
        else:
            left = 0
        if (i_trans < len(devide_points)):
            right = devide_points[i_trans]
        else:
            right = pred_arr.size

        stable = np.where(stable_frames[left: right] > 0)[0]
        # uniformly select 5 frames
        if (stable.size == 0):
            sample_idx = np.zeros(shape=(5,), dtype=int)
        else:
            sample_idx = np.linspace(0, stable.size - 1, 7, dtype=int)[1:6]
        _sample_frames = stable[sample_idx]
        sample_frames.append(_sample_frames + left)

    return {'devide_points': np.array(devide_points, dtype=int),
            'stable_frames': stable_frames,
            'sample_frames': np.array(sample_frames, dtype=int),
            'anomaly_regions': np.array(anomaly_regions)}


def get_t_model(device):
    model_params_path = os.path.join(ROOT, 'segment/transnetv2-pytorch-weights.pth')
    model = TransNetV2()
    state_dict = torch.load(model_params_path)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model
def predict_raw(frames: np.ndarray, model, device):
    """predict transition on a serious of frames"""
    assert len(frames.shape) == 5 and frames.shape[2:] == (27, 48, 3), \
        "[TransNetV2] Input shape must be [batch, frames, height, width, 3]."
    with torch.no_grad():
        frames = torch.tensor(frames, dtype=torch.uint8, device=device)
        logits, dict_ = model(frames)
        single_frame_pred = torch.sigmoid(logits)
        all_frames_pred = torch.sigmoid(dict_["many_hot"])

    return single_frame_pred.cpu(), all_frames_pred.cpu()

def predict_frames(frames: np.ndarray, model, device):
    """inference on the series of frames, cutting into overlapped windows"""
    assert len(frames.shape) == 4 and frames.shape[1:] == (27, 48, 3), \
        "[TransNetV2] Input shape must be [frames, height, width, 3]."

    def input_iterator():
        # return windows of size 100 where the first/last 25 frames are from the previous/next batch
        # the first and last window must be padded by copies of the first and last frame of the video
        no_padded_frames_start = 25
        no_padded_frames_end = 25 + 50 - (len(frames) % 50 if len(frames) % 50 != 0 else 50)  # 25 - 74

        start_frame = np.expand_dims(frames[0], 0)
        end_frame = np.expand_dims(frames[-1], 0)
        padded_inputs = np.concatenate(
            [start_frame] * no_padded_frames_start + [frames] + [end_frame] * no_padded_frames_end, 0
        )
        ptr = 0
        while ptr + 100 <= len(padded_inputs):
            out = padded_inputs[ptr:ptr + 100]
            ptr += 50
            yield out[np.newaxis]

    predictions = []
    for inp in input_iterator():
        single_frame_pred, all_frames_pred = predict_raw(inp, model, device)
        predictions.append((single_frame_pred.numpy()[0, 25:75, 0],
                            all_frames_pred.numpy()[0, 25:75, 0]))
    single_frame_pred = np.concatenate([single_ for single_, all_ in predictions])
    all_frames_pred = np.concatenate([all_ for single_, all_ in predictions])
    return single_frame_pred[:len(frames)], all_frames_pred[:len(frames)]  # remove extra padded frames

def get_stable_clips(video_path, model, device):
    video_stream, err = (ffmpeg
                         .input(video_path)
                         # .filter('trim', start=start_time, end=end_time)
                         .output("pipe:", format="rawvideo", pix_fmt="rgb24", s="48x27")
                         .run(capture_stdout=True, capture_stderr=True)
                         )
    video = np.frombuffer(video_stream, np.uint8).reshape([-1, 27, 48, 3])
    transition_preds = predict_frames(video, model, device)[0]
    cut_res = cut_one_video(transition_preds)
    result = segment_one(cut_res)
    s = result['left'][0]
    e = result['right'][-1]
    stable_clips = []
    for i in range(len(result['stable_left'])):
        stable_clips.append([result['stable_left'][i], result['stable_right'][i]])
    return stable_clips
##############

def cut_video_cv2(video_path, start_frame, end_frame, output_path):
    # Open the video file using OpenCV
    video = cv2.VideoCapture(video_path)
    # Get the video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate the start and end frames

    # Set the start frame as the current frame position
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Create a VideoWriter object to write the output video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Read video frames until the end frame
    while video.isOpened() and video.get(cv2.CAP_PROP_POS_FRAMES) <= end_frame:
        ret, frame = video.read()

        if ret:
            # Write the video frame to the output file
            output_video.write(frame)
        else:
            break

    # Release the video file and output video
    video.release()
    output_video.release()

def get_video_clip(video_path, stable_clips, clip_length, save_root, fps):
    clips = []
    for sc in stable_clips:
        i = sc[0]
        while i < sc[1]:
            c_start = i
            c_end = min(sc[1], i + clip_length)
            if (c_end - c_start) >= 0.06:
                clips.append({
                    'start': c_start,
                    'end': c_end
                })
            i += clip_length

    clips.sort(key=lambda c: c['start'])

    for id in range(len(clips)):
        save_path = os.path.join(save_root, f'{id}.mp4')
        clips[id]['id'] = id
        clips[id]['save_path'] = save_path
        # sub_video = video.subclip(clips[id]['start'], clips[id]['end'])
        # sub_video.write_videofile(save_path, verbose=False, logger=None)
        cut_video_cv2(video_path, int(clips[id]['start'] * fps), int(clips[id]['end'] * fps), save_path)
    return clips


def get_classes():
    """The activity categories"""
    classes = []
    with open(cate_p, 'r') as fp:
        for line in fp.readlines():
            classes.append(line.split('---')[1].strip('\n'))
    return np.array(classes)


def merge_clips(clips):
    clips.sort(key=lambda c: c['start'])
    end = clips[-1]['end']
    new_clips = []
    cur_clips = clips[0]
    for i in range(1, len(clips)):
        if clips[i]['class_id'] == cur_clips['class_id']:
            cur_clips['end'] = clips[i]['end']
        else:
            new_clips.append(cur_clips)
            cur_clips = clips[i]
    cur_clips['end'] = end
    new_clips.append(cur_clips)
    return new_clips


if __name__ == '__main__':
    args = get_args()
    set_seed(seed=args.seed, speed_up=False)
    input_path = args.input_path
    log_root = args.log_root

    name = os.path.basename(input_path)[:-len('.json')]
    output_video_tmp_root = f'/home2/xingyuan/animal_language/v2/tmp/{name}'
    output_path = f'/home2/xingyuan/animal_language/v2/out/context/mmaction/{name}.jsonlines'


    log_path = os.path.join(log_root, f'[mmaction_activity]_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log')

    assert os.path.exists(input_path), f'File or folder not found: {input_path}'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(log_root, exist_ok=True)
    if os.path.exists(output_video_tmp_root):
        shutil.rmtree(output_video_tmp_root)
    os.makedirs(output_video_tmp_root, exist_ok=True)

    log_init(log_path)
    logging.info(args)

    with open(input_path, 'r', encoding='utf-8') as file:
        input_datas = json.load(file)
    input_audios = []
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as file:
            output_datas = [json.loads(line.strip()) for line in file.readlines() if len(line.strip()) > 0]
        output_datas = [od['name'] for od in output_datas]
        output_datas.sort()
        for id in input_datas:
            ind = bisect.bisect_left(output_datas, id['name'])
            if not (0 <= ind < len(output_datas) and (output_datas[ind] == id['name'])):
                input_audios.append(id)
    else:
        input_audios = input_datas

    logging.info(f'Input data loaded. Will detection {len(input_audios)} audios (total={len(input_datas)}).')

    device = torch.device(f'cuda:{args.device}') if torch.cuda.is_available() and (args.device > -1) else torch.device('cpu')
    model = init_recognizer(config_path, checkpoint_path, device=device)
    t_model = get_t_model(device)
    model.eval()
    t_model.eval()
    model.to(device)
    t_model.to(device)
    logging.info(f'Model loaded(GPU={args.device}).')
    classes = get_classes()

    for audio in tqdm(input_audios):
        torch.cuda.empty_cache()
        if os.path.exists(output_video_tmp_root):
            shutil.rmtree(output_video_tmp_root)
        os.makedirs(output_video_tmp_root, exist_ok=True)

        video = cv2.VideoCapture(audio['video_path'])
        fps = video.get(cv2.CAP_PROP_FPS)
        duration = video.get(cv2.CAP_PROP_FRAME_COUNT) / fps

        dog_start = audio['start']
        dog_end = audio['end']
        start = max(0., dog_start - args.offset)
        end = min(duration, dog_end + args.offset)

        cut_video_cv2(audio['video_path'], int(start * fps), int(end * fps), os.path.join(output_video_tmp_root, 'v0.mp4'))
        # sub_video = video.subclip(start, end)
        # sub_video.write_videofile(os.path.join(output_video_tmp_root, 'v0.mp4'),
        #                           verbose=False, logger=None)

        try:
            stable_clips = get_stable_clips(os.path.join(output_video_tmp_root, 'v0.mp4'), model=t_model, device=device)
            for sc in stable_clips:
                sc[0] = start + (sc[0] / fps)
                sc[1] = start + (sc[1] / fps)
        except RuntimeError as re:
            if re.args[0] != 'Valid values are not enough for training.':
                raise re
            stable_clips = [[start, end]]

        clips = get_video_clip(video_path=audio['video_path'],
                               stable_clips=stable_clips,
                               clip_length=args.clip_length,
                               save_root=output_video_tmp_root,
                               fps=fps)

        for clip in clips:
            probs = inference_recognizer(model, clip['save_path']).pred_score
            y_hat = torch.argmax(probs).item()
            clip['class_id'] = y_hat
            clip['class_label'] = classes[y_hat]
        clips = [{"start": c['start'],
                  'end': c['end'],
                  'class_id': c['class_id'],
                  'class_label': c['class_label']} for c in clips]
        clips = merge_clips(clips)

        result = audio
        result['clips'] = clips
        with open(output_path, 'a+', encoding='utf-8') as file:
            file.write(json.dumps(result, ensure_ascii=False))
            file.write('\n')

    logging.info(f'Successfully detection {len(input_audios)} audios.')
