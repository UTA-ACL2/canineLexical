import os
import sys
import time

import librosa
import numpy as np

sys.path.append('../')
sys.path.append('../../')
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import argparse
import bisect
import json
import logging
import os
from datetime import datetime

import torch
from pytorch.models import *
from utils import config
from tqdm import tqdm

from util import set_seed, log_init

"""
conda activate /home/xingyuan/anaconda3/envs/animal_language_first_cut_panns
cd /bigdrive/xingyuan/animal_language2/code/first_cut/audioset_tagging_cnn/work

nohup python first_cut.py \
    --input_root /data/dog_age_classification/dataset/age/pre_process_dataset/dataset/videos \
    --output_path /data/dog_age_classification/dataset/age/pre_process_dataset/dataset/first_cut/dog_age_44k_thr0.05_offset1.jsonlines \
    --log_root /data/dog_age_classification/dataset/age/pre_process_dataset/logs/v2/log \
    --checkpoint_path /data/dog_age_classification/dataset/age/utils/first_cut/audioset_tagging_cnn/checkpoints/Cnn14_DecisionLevelMax_mAP=0.385.pth \
    --device 1 > /data/dog_age_classification/dataset/age/utils/first_cut/dog_age_44k_thr0.05_offset1.out 2>&1 &


"""

offset = 1.

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_root', type=str, required=True,
                        help='The root of audios')
    parser.add_argument('--output_path', type=str, required=True,
                        help='The path of output (JSONLINES FILE)')
    parser.add_argument('--log_root', type=str, required=False,
                        default='/data/xingyuan/animal_language/v2/log',
                        help='The root of log')

    parser.add_argument('--checkpoint_path', type=str, required=False,
                        default='/bigdrive/xingyuan/animal_language2/code/first_cut/audioset_tagging_cnn/checkpoints/Cnn14_DecisionLevelMax_mAP=0.385.pth',
                        help='The path of checkpoint')
    parser.add_argument('--threshold', type=float, required=False,
                        default=0.05,
                        help='The threshold of PANNs')
    parser.add_argument('--device', type=int, required=False,
                        default=0,
                        help='The number of GPU (-1 is CPU)')
    parser.add_argument('--seed', type=int, required=False,
                        default=42,
                        help='The seed')
    parser.add_argument('--audio_max_length', type=float, required=False,
                        default=60.0 * 60.0)  # second 60.0 * 60.0
    return parser.parse_args()


def move_data_to_device(x, device):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        return x

    return x.to(device)


def find_contiguous_regions(activity_array):
    """Find contiguous regions from bool valued numpy.array.
    Copy of https://dcase-repo.github.io/dcase_util/_modules/dcase_util/data/decisions.html#DecisionEncoder

    Reason is:
    1. This does not belong to a class necessarily
    2. Import DecisionEncoder requires sndfile over some other imports..which causes some problems on clusters

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


def sound_event_detection(waveform, device, frames_per_second, args, duration, start_time):
    waveform = waveform[None, :]    # (1, audio_length)
    waveform = move_data_to_device(waveform, device)

    # Forward
    with torch.no_grad():
        model.eval()
        batch_output_dict = model(waveform, None)

    framewise_output = batch_output_dict['framewise_output'].data.cpu().numpy()[0]
    """(time_steps, classes_num)"""

    dog_label_index = 74
    dog_framewise_output = framewise_output[:, dog_label_index]
    filtered_dog_framewise_output = np.where(dog_framewise_output >= args.threshold, 1, 0)
    continue_regions = find_contiguous_regions(filtered_dog_framewise_output)
    continue_regions = continue_regions / frames_per_second

    continue_regions[:, 0] -= offset
    continue_regions[:, 1] += offset
    continue_regions = np.maximum(continue_regions, 0.)
    continue_regions = np.minimum(continue_regions, duration)
    continue_regions += start_time
    clips = continue_regions.tolist()
    clips.sort(key=lambda c: c[0])
    return interval_union(clips=clips, interval_time=offset)


def interval_union(clips, interval_time=1.):
    if len(clips) < 2:
        return clips
    clips.sort(key=lambda c: c[0])
    start = clips[0][0]
    end = clips[0][1]

    new_clips = []
    for i in range(1, len(clips)):
        if (clips[i][0] - end) < interval_time:
            end = max(clips[i][1], end)
        else:
            new_clips.append([start, end])
            start = clips[i][0]
            end = clips[i][1]
    new_clips.append([start, end])
    return new_clips



if __name__ == '__main__':
    args = get_args()
    set_seed(seed=args.seed, speed_up=False)
    input_root = args.input_root
    output_path = args.output_path
    log_root = args.log_root
    checkpoint_path = args.checkpoint_path
    log_path = os.path.join(log_root, f'[first_cut]_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log')

    assert os.path.exists(input_root), f'File or folder not found: {input_root}'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(log_root, exist_ok=True)
    assert os.path.exists(checkpoint_path), f'File or folder not found: {checkpoint_path}'

    log_init(log_path)
    logging.info(args)

    input_datas = []
    for f in os.listdir(input_root):
        input_datas.append(os.path.join(input_root, f))
    input_datas = list(set(input_datas))
    input_audio_paths = []
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as file:
            output_datas = [json.loads(line.strip()) for line in file.readlines() if len(line.strip()) > 0]
        output_datas = [od['path'] for od in output_datas]
        output_datas.sort()
        for id in input_datas:
            ind = bisect.bisect_left(output_datas, id)
            if not (0 <= ind < len(output_datas) and (output_datas[ind] == id)):
                input_audio_paths.append(id)
    else:
        input_audio_paths = input_datas
    input_audio_paths = list(set(input_audio_paths))
    input_audio_paths.sort()
    if len(input_audio_paths) > 0:
        if input_audio_paths[0].split('.')[-1] not in ['npy', 'wav']:
            logging.error(f'Type error: Input file format error: {input_audio_paths[0].split(".")[-1]}.')
            exit()
    logging.info(f'Input data loaded. Will cut {len(input_audio_paths)} audios (total={len(input_datas)}).')

    sample_rate = 32000
    window_size = 1024
    hop_size = 320
    mel_bins = 64
    fmin = 50
    fmax = 14000
    model_type = 'Cnn14_DecisionLevelMax'
    device = torch.device(f'cuda:{args.device}') if torch.cuda.is_available() and (args.device > -1) else torch.device('cpu')

    classes_num = config.classes_num
    labels = config.labels
    frames_per_second = sample_rate // hop_size

    # Model
    Model = eval(model_type)
    model = Model(sample_rate=sample_rate, window_size=window_size,
                  hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax,
                  classes_num=classes_num)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)

    logging.info(f'Model loaded(GPU={device}).')

    audio_max_length = int(args.audio_max_length * sample_rate)
    cut_length = int(audio_max_length / 2)
    if len(input_audio_paths) > 0:
        if input_audio_paths[0].endswith('.npy'):
            logging.warning(f'The input data are .npy files, please verify if the sampling rate of the input data is {sample_rate}.')
    for path in tqdm(input_audio_paths):
        if path.endswith('.npy'):
            waveform = np.load(path)
        else:
            waveform, _ = librosa.core.load(path, sr=sample_rate, mono=True)

        ws = []
        if waveform.shape[0] > audio_max_length:
            start = 0
            while start < waveform.shape[0]:
                ws.append([waveform[start: start + audio_max_length], (start * 1.0) / sample_rate])
                start += cut_length
        else:
            ws = [[waveform, 0.0]]
        if ((waveform.shape[0] / sample_rate) - ws[-1][1]) < 0.2:
            ws.pop(-1)

        raw_clips = []
        for w in ws:
            clips = sound_event_detection(waveform=w[0],
                                          device=device,
                                          frames_per_second=frames_per_second,
                                          args=args,
                                          duration=w[0].shape[0] / sample_rate,
                                          start_time=w[1])
            raw_clips = raw_clips + clips

        raw_clips.sort(key=lambda c: c[0])
        raw_clips = interval_union(raw_clips, interval_time=offset)
        raw_clips.sort(key=lambda c: c[0])
        result_clips = []
        for i in range(len(raw_clips)):
            result_clips.append({
                'id': str(i).zfill(5),
                'start': raw_clips[i][0],
                'end': raw_clips[i][1]
            })

        result = {
            'name': os.path.basename(path)[:-len('.wav')],
            'path': path,
            'clips': result_clips
        }
        with open(output_path, 'a+', encoding='utf-8') as file:
            file.write(json.dumps(result, ensure_ascii=False))
            file.write('\n')

    logging.info(f'Successfully cut {len(input_audio_paths)} audios.')
