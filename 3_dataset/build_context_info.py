import json
import os
import shutil

import librosa
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
from tqdm import tqdm


def get_shiba_info():
    shiba_first_cut_path = '/home2/xingyuan/animal_language/v2/out/first_cut/shiba_thr0.05_offset1.jsonlines'
    shiba_second_cut_path = '/home2/xingyuan/animal_language/v2/out/second_cut/shiba_thr0.28_offset0.jsonlines'
    shiba_video_root = '/home/chunhao/AnimalLanguage/data/shibavideos_soft'
    save_path = '/home2/xingyuan/animal_language/v2/data/context/input_data/shiba_info.json'

    f_r_d = {}
    with open(shiba_first_cut_path, 'r', encoding='utf-8') as file:
        lines = [json.loads(line.strip()) for line in file.readlines() if len(line.strip()) > 0]
        for f_r in tqdm(lines):
            ytb = os.path.basename(os.path.dirname(f_r['path']))
            video_name = f_r['name'][:-len('.wav')]
            for clip in f_r['clips']:
                key = f'{ytb}_{video_name}_{clip["id"]}'
                assert key not in f_r_d.keys()
                assert os.path.exists(f_r['path'])
                f_r_d[key] = [clip['start'], clip['end'], f_r['path']]

    datas = []
    with open(shiba_second_cut_path, 'r', encoding='utf-8') as file:
        lines = [json.loads(line.strip()) for line in file.readlines() if len(line.strip()) > 0]
        for s_r in tqdm(lines):
            names = s_r['name'].split('_')
            ytb = '_'.join(names[0: -2])
            video_path = os.path.join(shiba_video_root, ytb, f'{names[-2]}.mp4')
            assert os.path.exists(video_path), video_path
            key = os.path.basename(s_r['path'])[:-len('.wav')]
            assert key in f_r_d.keys()
            f_start = f_r_d[key][0]
            for clip in s_r['clips']:
                datas.append({
                    'name': f'{key}_{clip["id"]}',
                    'video_path': video_path,
                    'audio_path': f_r_d[key][-1],
                    'start': f_start + clip['start'],
                    'end': f_start + clip['end']
                })

    with open(save_path, 'w+', encoding='utf-8') as file:
        json.dump(datas, file, indent=4, ensure_ascii=False)
    return datas


def get_chihuahua_info():
    chihuahua_first_cut_path = '/home2/xingyuan/animal_language/v2/out/first_cut/chihuahua_thr0.05_offset1.jsonlines'
    chihuahua_second_cut_path = '/home2/xingyuan/animal_language/v2/out/second_cut/chihuahua_thr0.28_offset0.jsonlines'
    chihuahua_video_root = '/home2/xingyuan/animal_language/data/videos/dog/Chihuahua'
    save_path = '/home2/xingyuan/animal_language/v2/data/context/input_data/chihuahua_info.json'

    f_r_d = {}
    with open(chihuahua_first_cut_path, 'r', encoding='utf-8') as file:
        lines = [json.loads(line.strip()) for line in file.readlines() if len(line.strip()) > 0]
        for f_r in tqdm(lines):
            name = f_r['name'][:-len('.wav')]
            video_path = os.path.join(chihuahua_video_root, f'{name}.mp4')
            assert os.path.exists(video_path), video_path
            for clip in f_r['clips']:
                key = f'{name}_{clip["id"]}'
                assert key not in f_r_d.keys()
                assert os.path.exists(f_r['path'])
                f_r_d[key] = [clip['start'], clip['end'], f_r['path'], video_path]

    datas = []
    with open(chihuahua_second_cut_path, 'r', encoding='utf-8') as file:
        lines = [json.loads(line.strip()) for line in file.readlines() if len(line.strip()) > 0]
        for s_r in tqdm(lines):
            key = s_r['name'][:-len('.wav')]
            assert key in f_r_d.keys()
            f_start = f_r_d[key][0]
            for clip in s_r['clips']:
                datas.append({
                    'name': f'{key}_{clip["id"]}',
                    'video_path': f_r_d[key][-1],
                    'audio_path': f_r_d[key][-2],
                    'start': f_start + clip['start'],
                    'end': f_start + clip['end']
                })

    with open(save_path, 'w+', encoding='utf-8') as file:
        json.dump(datas, file, indent=4, ensure_ascii=False)
    return datas


def get_dataset_info():
    dataset_first_cut_path = '/home2/xingyuan/animal_language/v2/out/first_cut/audioset_thr0.05_offset1.jsonlines'
    dataset_second_cut_path = '/home2/xingyuan/animal_language/v2/out/second_cut/audioset_thr0.28_offset0.jsonlines'
    dataset_video_root = '/home2/xingyuan/animal_language/audioset_videos'
    save_path = '/home2/xingyuan/animal_language/v2/data/context/input_data/dataset_info.json'

    f_r_d = {}
    with open(dataset_first_cut_path, 'r', encoding='utf-8') as file:
        lines = [json.loads(line.strip()) for line in file.readlines() if len(line.strip()) > 0]
        for f_r in tqdm(lines):
            name = f_r['name'][:-len('.wav')]
            video_path = os.path.join(dataset_video_root, f'{name}.mp4')
            if not os.path.exists(video_path):
                continue
            for clip in f_r['clips']:
                key = f'{name}_{clip["id"]}'
                assert key not in f_r_d.keys()
                assert os.path.exists(f_r['path'])
                f_r_d[key] = [clip['start'], clip['end'], f_r['path'], video_path]

    datas = []
    with open(dataset_second_cut_path, 'r', encoding='utf-8') as file:
        lines = [json.loads(line.strip()) for line in file.readlines() if len(line.strip()) > 0]
        for s_r in tqdm(lines):
            key = s_r['name'][:-len('.wav')]
            if key not in f_r_d.keys():
                continue
            f_start = f_r_d[key][0]
            for clip in s_r['clips']:
                datas.append({
                    'name': f'{key}_{clip["id"]}',
                    'video_path': f_r_d[key][-1],
                    'audio_path': f_r_d[key][-2],
                    'start': f_start + clip['start'],
                    'end': f_start + clip['end']
                })

    with open(save_path, 'w+', encoding='utf-8') as file:
        json.dump(datas, file, indent=4, ensure_ascii=False)
    return datas


def get_input_data():
    train_tsv_path = '/home2/xingyuan/animal_language/v2/code/audio_representation/hubert/work/all_thr0.28_offset0_rthr-0.7/tsv/train.tsv'
    chihuahua_info_path = '/home2/xingyuan/animal_language/v2/data/context/input_data/chihuahua_info.json'
    shiba_info_path = '/home2/xingyuan/animal_language/v2/data/context/input_data/shiba_info.json'
    dataset_info_path = '/home2/xingyuan/animal_language/v2/data/context/input_data/dataset_info.json'
    save_info_path = '/home2/xingyuan/animal_language/v2/data/context/input_data/train_info.json'

    def get_data(path):
        with open(path, 'r') as file:
            return json.load(file)

    datas = get_data(chihuahua_info_path) + get_data(shiba_info_path) + get_data(dataset_info_path)
    id2data = {}
    for data in tqdm(datas):
        assert data['name'] not in id2data.keys()
        id2data[data['name']] = data

    with open(train_tsv_path, 'r') as file:
        lines = [line.strip() for line in file.readlines() if len(line.strip()) > 0]
    count = 0
    results = []
    for line in tqdm(lines[1:]):
        id = line.split()[0][:-len('.wav')]
        if id in id2data.keys():
            results.append(id2data[id])
        else:
            count += 1
    with open(save_info_path, 'w+') as file:
        json.dump(results, file, indent=4, ensure_ascii=False)
    print(count)


def split_info(count):
    train_info_path = '/home2/xingyuan/animal_language/v2/data/context/input_data/train_info.json'
    save_root = '/home2/xingyuan/animal_language/v2/data/context/input_data/train_split'

    with open(train_info_path, 'r') as file:
        info_datas = json.load(file)
    step = int(len(info_datas) / count) + 1

    i = 0
    while i < len(info_datas):
        start = i
        end = min(len(info_datas), i + step)
        sub_data = info_datas[start: end]
        with open(os.path.join(save_root, f'train_{str(i)}.json'), 'w+') as file:
            json.dump(sub_data, file, indent=4, ensure_ascii=False)
        i = end

if __name__ == '__main__':
    root = '/data/xingyuan/animal_language/v2/out/separation/shiba51k_thr0.05_offset1'
    durations = []
    for file in tqdm(os.listdir(root)):
        wav, sr = librosa.load(os.path.join(root, file), sr=None, mono=True)
        durations.append(wav.shape[0] / sr)
    print(f'total duration:\t{sum(durations) / 3600.}')
    print(f'min duration:\t{min(durations)}')
    print(f'mean duration:\t{np.mean(durations)}')
    print(f'max duration:\t{max(durations)}')
    print(f'count:\t{len(durations)}')
