import argparse
import os
import random

import librosa
import matplotlib.pyplot as plt
import numpy as np
import textgrid
from torch.utils.data import Dataset
from tqdm import tqdm


def pad_truncate_sequence(x, max_len):
    ori_len = x.shape[0]
    if ori_len < max_len:
        return np.concatenate((x, np.zeros(int(max_len - ori_len))), axis=0), ori_len
    else:
        return x[0: max_len], ori_len


def slide_cut(arr, window_size, stride):
    tail_len = window_size - stride
    duration = arr.shape[0]

    if duration < window_size:
        return [[pad_truncate_sequence(arr, window_size)[0], int(duration)]]

    count = (duration - tail_len) // stride
    remainder_len = (duration - tail_len) % stride

    if remainder_len != 0:
        count += 1
        all_len = int(tail_len + count * stride)
        arr, _ = pad_truncate_sequence(arr, all_len)

    clips = []
    for i in range(int(count)):
        start = int(i * stride)
        end = int(start + window_size)
        clips.append([arr[start: end], int(window_size)])
    if remainder_len != 0:
        clips[-1][1] = int(remainder_len)
    return clips


class DogDataset(Dataset):
    def __init__(self, args, mode='train'):
        assert mode in ['train', 'test'], f'Error mode: {mode}'
        self.mode = mode
        self.args = args
        self.labels_root = args.train_labels_root if mode == 'train' else args.test_labels_root
        self.datas = []
        self.positive_labels = args.train_positive_labels if mode == 'train' else args.test_positive_labels

        self.audios = []
        for tg_file in os.listdir(self.labels_root):
            name = tg_file[: -len('.TextGrid')]
            wav, _ = librosa.load(os.path.join(args.audios_root, f'{name}.wav'), sr=self.args.sr, mono=True)
            duration = wav.shape[0] / self.args.sr
            tg = textgrid.TextGrid.fromFile(os.path.join(self.labels_root, tg_file))

            clips = []
            for dog_tier in tg.getList("dog"):
                for interval in dog_tier.intervals:
                    clips.append([[interval.minTime, min(duration, interval.maxTime)], 1 if interval.mark in self.positive_labels else 0])

            self.audios.append({
                'name': name,
                'wav': wav,
                'duration': duration,
                'clips': clips
            })

        self.preprocessing()

        # if self.mode == 'train' and args.train_filter_flag:
        #     self.train_filter2()

    def __getitem__(self, index):
        return self.datas[index]

    def __len__(self):
        return len(self.datas)

    def train_filter(self, max_times=1.5):
        new_datas = []
        name2negs = {}
        for data in self.datas:
            if np.sum(data['target']) > 0.:
                new_datas.append(data)
            else:
                if data['name'] in name2negs.keys():
                    name2negs[data['name']].append(data)
                else:
                    name2negs[data['name']] = [data]
        for name in name2negs.keys():
            random.shuffle(name2negs[name])

        negs = []
        for name in name2negs.keys():
            negs.append(name2negs[name][0])
        negs_count = int(len(new_datas) * max_times + 1) - len(negs)

        ind = 1
        while negs_count > 0:
            for name in name2negs.keys():
                if ind < len(name2negs[name]) and (negs_count > 0):
                    negs.append(name2negs[name][ind])
                    negs_count -= 1
            ind += 1
        new_datas.extend(negs)
        random.shuffle(new_datas)
        self.datas = new_datas

    def train_filter2(self):
        print(len(self.datas))
        self.datas = [data for data in self.datas if np.sum(data['target']) > 0.]
        print(len(self.datas))
        exit()


    def preprocessing(self):
        if self.mode == 'train':
            for audio in self.audios:
                target = np.zeros(audio['wav'].shape[0])
                for clip in audio['clips']:
                    if clip[1] == 1:
                        target[int(clip[0][0] * self.args.sr): int(clip[0][1] * self.args.sr)] = 1.

                audio_clips = slide_cut(audio['wav'], window_size=self.args.data_window_size * self.args.sr, stride=self.args.data_stride * self.args.sr)
                target_clips = slide_cut(target, window_size=self.args.data_window_size * self.args.sr, stride=self.args.data_stride * self.args.sr)
                assert len(audio_clips) == len(target_clips)
                assert audio_clips[-1][1] == target_clips[-1][1]
                for i in range(len(audio_clips)):
                    self.datas.append({
                        'name': audio['name'],
                        'waveform': audio_clips[i][0],
                        'target': target_clips[i][0],
                        'origin_length': audio_clips[i][1]
                    })
        else:
            for audio in self.audios:
                target = np.zeros(audio['wav'].shape[0])
                for clip in audio['clips']:
                    if clip[1] == 1:
                        target[int(clip[0][0] * self.args.sr): int(clip[0][1] * self.args.sr)] = 1.

                self.datas.append({
                    'name': audio['name'],
                    'waveform': audio['wav'],
                    'target': target
                })


ground_truth = {}
durations = {}
def get_seds_ground_truth(label_root, audios_root, sr, test_positive_labels):
    if len(ground_truth.keys()) > 0 and (len(durations.keys()) > 0):
        return ground_truth, durations

    for tg_file in os.listdir(label_root):
        id = tg_file[:-len('.TextGrid')]
        tg = textgrid.TextGrid.fromFile(os.path.join(label_root, tg_file))
        wav, _ = librosa.load(os.path.join(audios_root, f'{id}.wav'), sr=sr, mono=True)

        durations[id] = wav.shape[0] / sr
        clips = []
        for dog_tier in tg.getList("dog"):
            for interval in dog_tier.intervals:
                if interval.mark in test_positive_labels:
                    clips.append((interval.minTime, min(interval.maxTime - 0.001, durations[id]), 'dog'))
        ground_truth[id] = clips
    return ground_truth, durations


def show_data(root, train_positive_labels, test_positive_labels):
    train_labels = os.listdir(os.path.join(root, 'train_labels'))
    valid_labels = os.listdir(os.path.join(root, 'valid_labels'))
    for vl in valid_labels:
        assert vl not in train_labels

    def show_labels(label_root, positive_labels):
        interval_list = []
        for label_file in os.listdir(label_root):
            tg = textgrid.TextGrid.fromFile(os.path.join(label_root, label_file))
            for dog_tier in tg.getList('dog'):
                for interval in dog_tier.intervals:
                    if interval.mark in positive_labels:
                        interval_list.append(interval.maxTime - interval.minTime)
        return interval_list

    train_interval_list = show_labels(label_root=os.path.join(root, 'train_labels'), positive_labels=train_positive_labels)
    valid_interval_list = show_labels(label_root=os.path.join(root, 'valid_labels'), positive_labels=test_positive_labels)
    all_interval_list = train_interval_list + valid_interval_list
    print(f"train: {(min(train_interval_list), np.mean(train_interval_list), max(train_interval_list), len(train_interval_list))}")
    print(f"valid: {(min(valid_interval_list), np.mean(valid_interval_list), max(valid_interval_list), len(valid_interval_list))}")
    print(f"all: {(min(all_interval_list), np.mean(all_interval_list), max(all_interval_list))}")

    audios = []
    for audio_file in tqdm(os.listdir(os.path.join(root, 'audios'))):
        name = audio_file[: -len('.wav')]
        if f'{name}.TextGrid' in valid_labels: continue
        wav, sr = librosa.load(os.path.join(root, 'audios', audio_file), mono=True)
        duration = wav.shape[0] / sr
        # if duration > 20.: continue
        audios.append(duration)
    print(f"audios: {(min(audios), np.mean(audios), max(audios), len(audios))}")

    audios.sort()
    # for i in range(len(audios)):
    #     if audios[i] > 30.:
    #         audios[i] = 30.
    plt.hist(audios)
    plt.show()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, required=False,
                        default=3,
                        help='The number of GPU (-1 is CPU)')
    parser.add_argument('--seed', type=int, required=False,
                        default=42,
                        help='The seed')

    # train
    parser.add_argument('--save_root', type=str, required=False,
                        default='/bigdrive/xingyuan/v2/out/second_cut/panns/alldata_thr0.05_offset1_0-0.01_0.9')
    parser.add_argument('--learning_rate', type=float, required=False,
                        default=1e-3)
    parser.add_argument('--steps', type=int, required=False,
                        default=999999)
    parser.add_argument('--epochs', type=int, required=False,
                        default=200)
    parser.add_argument('--test_by_step', type=int, required=False,
                        default=20)
    parser.add_argument('--train_positive_labels', type=str, required=False,
                        default='dog,dog_dog,dog_noise')
    parser.add_argument('--test_positive_labels', type=str, required=False,
                        default='dog,dog_dog,dog_noise')

    # dataset
    parser.add_argument('--sr', type=int, required=False,
                        default=32000)
    parser.add_argument('--data_window_size', type=float, required=False,
                        default=10.0)
    parser.add_argument('--data_stride', type=float, required=False,
                        default=5.0)
    parser.add_argument('--audios_root', type=str, required=False,
                        default='/home/xingyuan/animal_language2/data/train_second_cut/alldata_thr0.05_offset1_0-0.01_0.9/audios')
    parser.add_argument('--train_labels_root', type=str, required=False,
                        default='/home/xingyuan/animal_language2/data/train_second_cut/alldata_thr0.05_offset1_0-0.01_0.9/train_labels')
    parser.add_argument('--train_batch_size', type=int, required=False,
                        default=16)
    parser.add_argument('--train_num_workers', type=int, required=False,
                        default=8)
    parser.add_argument('--train_filter_flag', required=False, action='store_true',
                        default=False)

    parser.add_argument('--test_labels_root', type=str, required=False,
                        default='/home/xingyuan/animal_language2/data/train_second_cut/alldata_thr0.05_offset1_0-0.01_0.9/valid_labels')
    parser.add_argument('--test_batch_size', type=int, required=False,
                        default=1)
    parser.add_argument('--test_num_workers', type=int, required=False,
                        default=8)
    return parser.parse_args()

"""
1451-16-90.68-4.534
267
2592-32-81-18

5027-64-78.54-17

"""
if __name__ == '__main__':
    args = get_args()
    args.train_positive_labels = args.train_positive_labels.split(',')
    args.test_positive_labels = args.test_positive_labels.split(',')
    dataset = DogDataset(args, mode='test')
    target_l = 0
    all_l = 0
    # for one in dataset:
    #     all_l += one['origin_length']
    #     target_l += np.sum(one['target'][0: one['origin_length']])

    for one in dataset:
        all_l += len(one['target'])
        target_l += np.sum(one['target'])
    print(target_l)
    print(all_l)
    print(target_l / all_l)


    # root = '/home/xingyuan/animal_language2/data/train_second_cut/alldata_thr0.05_offset1_0-0.01_0.9'
    # show_data(root=root,
    #           train_positive_labels=['dog', 'dog_dog', 'dog_noise'],
    #           test_positive_labels=['dog', 'dog_dog', 'dog_noise'])

    # root = '/home/xingyuan/animal_language2/data/train_second_cut/alldata_thr0.05_offset1_0-0.01_0.9/train_labels'
    # for tg_file in os.listdir(root):
    #     tg = textgrid.TextGrid.fromFile(os.path.join(root, tg_file))
    #     ok = []
    #     for dog_tier in tg.getList('dog'):
    #         for tier_interval in dog_tier.intervals:
    #             if tier_interval.mark is not None and (tier_interval.mark != ''):
    #                 ok.append(tier_interval)
    #     if len(ok) < 1:
    #         print(tg_file)

    # root = '/home2/xingyuan/animal_language/v2/data/train_second_cut/alldata_thr0.05_offset1_0-0.01_0.9'
    # show_data(root=root,
    #           train_positive_labels=['dog', 'dog_dog', 'dog_noise'],
    #           test_positive_labels=['dog', 'dog_dog', 'dog_noise'])

    # root = '/home2/xingyuan/animal_language/pipeline/sentence/panns_finetuning/dataset/labels'
    # interval_list = []
    # audios = []
    # for label_file in os.listdir(root):
    #     tg = textgrid.TextGrid.fromFile(os.path.join(root, label_file))
    #     audios.append(tg.maxTime - tg.minTime)
    #     for dog_tier in tg.tiers:
    #         for interval in dog_tier.intervals:
    #             if interval.mark == '1':
    #                 interval_list.append(interval.maxTime - interval.minTime)
    # print((min(interval_list), np.mean(interval_list), max(interval_list)))
    # print(f"audios: {(min(audios), np.mean(audios), max(audios), len(audios))}")

