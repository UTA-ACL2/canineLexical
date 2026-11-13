import csv
import json
import os
import sys

import librosa
import numpy
import soundfile
from sklearn.cluster import MiniBatchKMeans
import torch.nn.functional as F

sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
sys.path.append('../../../../')
sys.path.append('../../../../../')
import random
import shutil
import fairseq

import torch
from tqdm import tqdm
from fairseq.data.audio.audio_utils import get_features_or_waveform

def split_dataset(audios_root,
                  save_tsv_root,
                  valid_ratio=0.1):
    random.seed(0)
    audios = [f for f in os.listdir(audios_root)]
    audios.sort()
    random.seed(0)
    random.shuffle(audios)

    valid_count = int(len(audios) * valid_ratio) + 1
    valid_audios = audios[0: valid_count]
    train_audios = audios[valid_count: ]
    print(len(valid_audios))
    print(len(train_audios))

    if os.path.exists(save_tsv_root): shutil.rmtree(save_tsv_root)
    os.makedirs(save_tsv_root)

    def get_results(cur_audios):
        sample_rate = 16000
        results = []
        for r in tqdm(cur_audios):
            assert os.path.exists(f'{audios_root}/{r}')
            wav = get_features_or_waveform(f'{audios_root}/{r}',
                                           need_waveform=True,
                                           use_sample_rate=sample_rate)
            if wav.ndim == 2:
                wav = wav.mean(-1)
            assert wav.ndim == 1, wav.ndim
            results.append([r, len(wav)])
        return results

    train_results = get_results(train_audios)
    valid_results = get_results(valid_audios)
    train_results.insert(0, [audios_root])
    valid_results.insert(0, [audios_root])

    with open(os.path.join(save_tsv_root, 'train.tsv'), 'w+', encoding='utf-8', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerows(train_results)

    with open(os.path.join(save_tsv_root, 'valid.tsv'), 'w+', encoding='utf-8', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerows(valid_results)



if __name__ == '__main__':
    audios_root = '/data/xingyuan/animal_language/v2/out/fairseq/six_balance/data/150-5000-4000/all'
    save_tsv_root = '/data/xingyuan/animal_language/v2/out/fairseq/six_balance/tsv'
    split_dataset(audios_root,
                  save_tsv_root,
                  valid_ratio=0.1)

