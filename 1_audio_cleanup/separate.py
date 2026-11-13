import sys
import os

import librosa
import numpy as np
from tqdm import tqdm

sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')
sys.path.insert(1, os.path.join(sys.path[0], '../'))

import argparse
import json
import logging

from datetime import datetime

import torch

from util import set_seed, log_init
from scipy.io.wavfile import write
from pipeline import *


"""
conda activate animal_language_separation_audiosep
cd /home/xingyuan/animal_language2/code/separation/AudioSep

nohup python ./work/separate.py \
             --input_path /bigdrive/xingyuan/v2/out/first_cut/shiba_thr0.05_offset1.jsonlines \
             --output_root /bigdrive/xingyuan/v2/out/separation/shiba_thr0.05_offset1 \
             --log_root /bigdrive/xingyuan/v2/log \
             --checkpoint_path /home/xingyuan/animal_language2/code/separation/AudioSep/checkpoint/audiosep_base_4M_steps.ckpt \
             --device 0 \
             --overwrite > /bigdrive/xingyuan/v2/out/separation/separation_shiba_thr0.05_offset1.out 2>&1 &

"""
"""
cd /bigdrive/xingyuan/animal_language2/code/separation/AudioSep/
conda activate /home/xingyuan/anaconda3/envs/animal_language_separation_audiosep
nohup python ./work/separate.py \
             --input_path /data/dog_age_classification/dog_gender/preprocess_dataset/first_cut/dog_age_60k_thr0.05_offset1.jsonlines \
             --output_root /data/dog_age_classification/dog_gender/preprocess_dataset/first_cut/separation/dog_age_60k_thr0.05_offset1 \
             --log_root /data/dog_age_classification/dog_gender/logs \
             --checkpoint_path /bigdrive/xingyuan/animal_language2/code/separation/AudioSep/checkpoint/audiosep_base_4M_steps.ckpt \
             --device 1 \
             --overwrite > /data/dog_age_classification/dog_gender/preprocess_dataset/first_cut/a.out 2>&1 &


"""

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True,
                        help='The path of input (JSONLINES FILE)')
    parser.add_argument('--output_root', type=str, required=True,
                        help='The root of output')
    parser.add_argument('--log_root', type=str, required=False,
                        default='/data/xingyuan/animal_language/v2/log',
                        help='The root of log')

    parser.add_argument('--checkpoint_path', type=str, required=False,
                        default='/bigdrive/xingyuan/animal_language2/code/separation/AudioSep/checkpoint/audiosep_base_4M_steps.ckpt',
                        help='The path of checkpoint')
    parser.add_argument('--device', type=int, required=False,
                        default=0,
                        help='The number of GPU (-1 is CPU)')
    parser.add_argument('--seed', type=int, required=False,
                        default=42,
                        help='The seed')
    parser.add_argument('--chunk_sec', type=float, required=False,
                        default=128.,
                        help='The chunk_sec')
    parser.add_argument('--text', type=str, required=False,
                        default='Dog',
                        help='text')
    parser.add_argument('--overwrite', required=False, action='store_true',
                        default=False,
                        help='Whether or not to overwrite')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    set_seed(seed=args.seed, speed_up=False)
    input_path = args.input_path
    output_root = args.output_root
    log_root = args.log_root
    checkpoint_path = args.checkpoint_path
    log_path = os.path.join(log_root, f'[separation]_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log')

    assert os.path.exists(input_path), f'File or folder not found: {input_path}'
    os.makedirs(output_root, exist_ok=True)
    os.makedirs(log_root, exist_ok=True)
    assert os.path.exists(checkpoint_path), f'File or folder not found: {checkpoint_path}'

    log_init(log_path)
    logging.info(args)

    with open(input_path, 'r', encoding='utf-8') as file:
        raw_datas = [json.loads(line.strip()) for line in file.readlines() if len(line.strip()) > 0]
    input_datas = []
    # existing_files = set()
    # if os.path.exists(output_root) and os.path.isdir(output_root):
    #     for filename in os.listdir(output_root):
    #         if filename.endswith('.wav'):
    #             existing_files.add(f'{filename}')

    for rd in raw_datas:
        for clip in rd['clips']:
            output_path = os.path.join(output_root, f'{rd["name"]}+{clip["id"]}.wav')
            # logging.info(f"{output_path}, {output_path in existing_files}")
            if (not os.path.exists(output_path)) or args.overwrite:
                logging.info(f"Not exist: {output_path}")
                
                input_datas.append({
                    'input_path': rd['path'],
                    'start': clip['start'],
                    'duration': clip['end'] - clip['start'],
                    'output_path': output_path
                })
    if input_datas[0]['input_path'].split('.')[-1] not in ['npy', 'wav','mp4']:
        logging.error(f'Type error: Input file format error: {input_datas[0]["input_path"].split(".")[-1]}.')
        exit()

    input_datas.sort(key=lambda id: -id['duration'])
    logging.info(f'Max duration: {input_datas[0]["duration"]}')
    logging.info(f'Input data loaded. Will separate {len(input_datas)} audios.')

    device = torch.device(f'cuda:{args.device}') if torch.cuda.is_available() and (args.device > -1) else torch.device('cpu')
    model = build_audiosep(config_yaml='/bigdrive/xingyuan/animal_language2/code/separation/AudioSep/config/audiosep_base.yaml',
                           checkpoint_path=checkpoint_path,
                           device=device)
    text = args.text
    with torch.no_grad():
        text = [text]

        conditions = model.query_encoder.get_query_embed(
            modality='text',
            text=text,
            device=device
        )
    logging.info(f'Model loaded(GPU={args.device}).')

    if input_datas[0]['input_path'].endswith('.npy'):
        logging.warning(f'The input data are .npy files, please verify if the sampling rate of the input data is 32000.')
    use_chunk = False
    for data in tqdm(input_datas):
        torch.cuda.empty_cache()
        if data['input_path'].endswith('.npy'):
            mixture = np.load(data['input_path'])
            s = int(data['start'] * 32000)
            e = int(s + data['duration'] * 32000) + 1
            mixture = mixture[s: e]
        else:
            mixture, fs = librosa.load(data['input_path'],
                                       sr=32000,
                                       mono=True,
                                       offset=data['start'],
                                       duration=data['duration'])

        use_chunk = (data['duration'] >= args.chunk_sec)
        # print(f"{data['duration']}: {use_chunk}")

        with torch.no_grad():
            input_dict = {
                "mixture": torch.Tensor(mixture)[None, None, :].to(device),
                "condition": conditions,
            }

            if use_chunk:
                sep_segment = model.ss_model.chunk_inference(input_dict)
                sep_segment = np.squeeze(sep_segment)
            else:
                sep_segment = model.ss_model(input_dict)["waveform"]
                sep_segment = sep_segment.squeeze(0).squeeze(0).data.cpu().numpy()

            write(data['output_path'], 32000, np.round(sep_segment * 32767).astype(np.int16))

    logging.info(f'Successfully sep {len(input_datas)}(total={len(input_datas)}) clips.')

