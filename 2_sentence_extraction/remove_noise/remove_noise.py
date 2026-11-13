import argparse
import json
import logging
import os
import shutil
import sys
from datetime import datetime

import librosa
import numpy as np
import torch
from pydub import AudioSegment
from tqdm import tqdm

sys.path.append('../')
sys.path.append('../../')
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))

from util import set_seed, log_init
from pytorch.models import *
from utils import config


"""
conda activate animal_language_first_cut_panns
cd /home/xingyuan/animal_language2/code/remove_noise/audioset_tagging_cnn-master/work

python ./remove_noise.py \
       --input_path /bigdrive/xingyuan/v2/out/second_cut/shiba_thr0.28_offset0.jsonlines \
       --output_root /bigdrive/xingyuan/v2/out/remove_noise/shiba_thr-0.7 \
       --log_root /bigdrive/xingyuan/v2/log \
       --checkpoint_path /home/xingyuan/animal_language2/code/remove_noise/audioset_tagging_cnn-master/checkpoints/Cnn14_16k_mAP=0.438.pth \
       --topk 10 \
       --threshold -0.7 \
       --device 0

python ./remove_noise.py \
       --input_path /bigdrive/xingyuan/v2/out/second_cut/shiba_thr0.28_offset0.jsonlines \
       --output_root /bigdrive/xingyuan/v2/out/remove_noise/shiba_thr-0.5 \
       --log_root /bigdrive/xingyuan/v2/log \
       --checkpoint_path /home/xingyuan/animal_language2/code/remove_noise/audioset_tagging_cnn-master/checkpoints/Cnn14_16k_mAP=0.438.pth \
       --topk 10 \
       --threshold -0.5 \
       --device 0

python ./remove_noise.py \
       --input_path /bigdrive/xingyuan/v2/out/second_cut/shiba_thr0.28_offset0.jsonlines \
       --output_root /bigdrive/xingyuan/v2/out/remove_noise/shiba_thr-0.3 \
       --log_root /bigdrive/xingyuan/v2/log \
       --checkpoint_path /home/xingyuan/animal_language2/code/remove_noise/audioset_tagging_cnn-master/checkpoints/Cnn14_16k_mAP=0.438.pth \
       --topk 10 \
       --threshold -0.3 \
       --device 1

"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True,
                        help='The path of input (TXT FILE)')
    parser.add_argument('--output_root', type=str, required=True,
                        help='The root of output')
    parser.add_argument('--log_root', type=str, required=True,
                        help='The root of log')

    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='The path of checkpoint')
    parser.add_argument('--device', type=int, required=False,
                        default=3,
                        help='The number of GPU (-1 is CPU)')
    parser.add_argument('--topk', type=int, required=False,
                        default=10,
                        help='The topk')
    parser.add_argument('--threshold', type=float, required=True,
                        default=-1.0,
                        help='The threshold')
    parser.add_argument('--seed', type=int, required=False,
                        default=42,
                        help='The seed')
    return parser.parse_args()


def move_data_to_device(x, device):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        return x

    return x.to(device)


def check(scores, thr):
    exlabels = [
        'Animal',
        'Domestic animals, pets',
        'Canidae, dogs, wolves',
        'Whimper (dog)',
        'Bow-wow',
        'Yip',
        'Bark',
        'Growling',
        'Howl',
        # 'Inside, small room',
        # 'Outside, rural or natural'
    ]
    scores.sort(key=lambda s: -s['score'])
    # 过滤分数小于0.05的标签
    scores = [s for s in scores if s['score'] > 0.05]

    # 如果Speech, Music在top5，则过滤
    for s in scores:
        if s['label'] == 'Speech' or (s['label'] == 'Music'):
            return False

    # 获取dog标签的分数，如果分数小于0.1则过滤
    dog_score = -1.0
    for s in scores:
        if s['label'] == 'Dog':
            dog_score = s['score']
            break
    if dog_score < 0.1:
        return False

    # 排除分数和dog的距离超过thr的情况
    for s in scores:
        if s['label'] == 'Dog':
            continue
        if ((s['score'] - dog_score) > thr) and (s['label'] not in exlabels):
            return False

    return True


if __name__ == '__main__':
    args = get_args()
    set_seed(seed=args.seed, speed_up=False)
    input_path = args.input_path
    output_root = args.output_root
    log_root = args.log_root
    checkpoint_path = args.checkpoint_path
    log_path = os.path.join(log_root, f'[remove_noise]_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log')

    if os.path.exists(output_root):
        shutil.rmtree(output_root)
    assert os.path.exists(input_path), f'File or folder not found: {input_path}'
    os.makedirs(output_root, exist_ok=True)
    os.makedirs(log_root, exist_ok=True)
    assert os.path.exists(checkpoint_path), f'File or folder not found: {checkpoint_path}'

    log_init(log_path)
    logging.info(args)

    with open(input_path, 'r', encoding='utf-8') as file:
        input_audios = [json.loads(line.strip()) for line in file.readlines() if len(line.strip()) > 0]
    logging.info(f'Input data loaded. Will check {len(input_audios)} audios (total={len(input_audios)}).')

    model_type = 'Cnn14_16k'
    sample_rate = 16000
    window_size = 512
    hop_size = 160
    mel_bins = 64
    fmin = 50
    fmax = 8000
    classes_num = config.classes_num
    labels = config.labels
    device = torch.device(f'cuda:{args.device}') if torch.cuda.is_available() and (args.device > -1) else torch.device('cpu')
    frames_per_second = sample_rate // hop_size

    Model = eval(model_type)
    model = Model(sample_rate=sample_rate,
                  window_size=window_size,
                  hop_size=hop_size,
                  mel_bins=mel_bins,
                  fmin=fmin,
                  fmax=fmax,
                  classes_num=classes_num)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    logging.info(f'Model loaded(GPU={args.device}).')

    for audio in tqdm(input_audios):
        name = audio['name'][: -len('.wav')]
        audio_wav = AudioSegment.from_wav(audio['path'])
        for clip in audio['clips']:
            save_path = os.path.join(output_root, f"{name}_{clip['id']}.wav")
            if os.path.exists(save_path): continue

            waveform, _ = librosa.load(audio['path'], sr=sample_rate, mono=True,
                                       offset=clip['start'],
                                       duration=clip['end'] - clip['start'])
            waveform = waveform[None, :]
            waveform = move_data_to_device(waveform, device)

            with torch.no_grad():
                model.eval()
                batch_output_dict = model(waveform, None)

            clipwise_output = batch_output_dict['clipwise_output'].data.cpu().numpy()[0]
            sorted_indexes = np.argsort(clipwise_output)[::-1]
            scores = []
            for k in range(args.topk):
                scores.append({
                    'label': np.array(labels)[sorted_indexes[k]],
                    'score': float(clipwise_output[sorted_indexes[k]])
                })

            if check(scores, thr=args.threshold):
                clip_wav = audio_wav[clip['start'] * 1000.: clip['end'] * 1000.]
                clip_wav.export(save_path)

