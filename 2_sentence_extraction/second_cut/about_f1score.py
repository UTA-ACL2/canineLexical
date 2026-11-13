import os
import sys

from torch.utils.data import DataLoader, ConcatDataset

sys.path.append('../')
sys.path.append('../../')
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))

import torch
from pytorch.models import *
# from dataset import DogDataset

from util import set_seed

import pandas as pd
from dataset2 import DogDataset
from sed_scores_eval import *
from evaluate2 import *


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
                        default=10)
    parser.add_argument('--train_positive_labels', type=str, required=False,
                        default='dog,dog_dog,dog_noise')
    parser.add_argument('--test_positive_labels', type=str, required=False,
                        default='dog,dog_dog')

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


def get_model():
    checkpoint_path = '/home/xingyuan/animal_language2/code/second_cut/audioset_tagging_cnn-master/work/checkpoint_51_2910.pth'
    sample_rate = 32000
    window_size = 1024
    hop_size = 320
    mel_bins = 64
    fmin = 50
    fmax = 14000
    model_type = 'Cnn14_DecisionLevelMax'
    device = torch.device(f'cpu')

    classes_num = 1
    frames_per_second = sample_rate // hop_size

    # Model
    Model = eval(model_type)
    model = Model(sample_rate=sample_rate, window_size=window_size,
                  hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax,
                  classes_num=classes_num)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    print(f'Model loaded(GPU={device}).')
    return model, frames_per_second

ground_truth = {}
durations = {}
def get_seds_ground_truth(label_audios, label_root, audios_root, sr, test_positive_labels):
    if len(ground_truth.keys()) > 0 and (len(durations.keys()) > 0):
        return ground_truth, durations

    for la in label_audios:
        tg_file = os.path.basename(la['label_path'])
        id = tg_file[:-len('.TextGrid')]
        tg = textgrid.TextGrid.fromFile(la['label_path'])
        wav, _ = librosa.load(la['audio_path'], sr=sr, mono=True)

        durations[id] = wav.shape[0] / sr
        clips = []
        for dog_tier in tg.getList("dog"):
            for interval in dog_tier.intervals:
                if interval.mark in test_positive_labels:
                    clips.append((interval.minTime, min(interval.maxTime - 0.001, durations[id]), 'dog'))
        ground_truth[id] = clips
    return ground_truth, durations

def myeval(*, label_audios, test_results, label_root, frames_per_second, audios_root, sr, test_positive_labels):
    # segment指标的粒度，即最小单元长度
    segment_length = 0.10
    num_jobs = 3
    # 开始、结束位置可容忍范围
    collar = 0.10
    # 允许的最小recall
    best_f1_min_recall = 0.8
    # 交集/交集+pred
    dtc_threshold = 0.25
    # 交集/交集+gt
    gtc_threshold = 0.8

    seds_pred = {}
    for id in test_results.keys():
        clips = []
        start = 0
        end = 1
        for fwi in range(1, len(test_results[id])):
            if abs(test_results[id][fwi] - test_results[id][fwi - 1]) < 1e-6:
                end += 1
            else:
                clips.append({
                    'onset': start / frames_per_second,
                    'offset': end / frames_per_second,
                    'dog': np.mean(test_results[id][start: end])
                })
                start = fwi
                end = fwi + 1
        clips.append({
            'onset': start / frames_per_second,
            'offset': end / frames_per_second,
            'dog': np.mean(test_results[id][start: end])
        })
        seds_pred[id] = pd.DataFrame(clips)

    results = {
        'segment_based': {},
        'collar_based': {},
        'intersection_based': {}
    }
    seds_ground_truth, seds_durations = get_seds_ground_truth(label_audios=label_audios, label_root=label_root, audios_root=audios_root, sr=sr, test_positive_labels=test_positive_labels)


    # segment based best f1
    best_f1, _, __, best_threshold, ___ = segment_based.best_fscore(
        scores=seds_pred,
        ground_truth=seds_ground_truth,
        audio_durations=seds_durations,
        min_recall=best_f1_min_recall,
        segment_length=segment_length,
        num_jobs=num_jobs
    )
    results['segment_based']['best_f1_score'] = best_f1['dog']
    results['segment_based']['best_f1_threshold'] = best_threshold['dog']

    # intersection based best f1
    best_f1, _, __, best_threshold, ___ = intersection_based.best_fscore(
        scores=seds_pred,
        ground_truth=seds_ground_truth,
        dtc_threshold=dtc_threshold,
        gtc_threshold=gtc_threshold,
        min_recall=best_f1_min_recall,
        num_jobs=num_jobs
    )
    results['intersection_based']['best_f1_score'] = best_f1['dog']
    results['intersection_based']['best_f1_threshold'] = best_threshold['dog']

    return results

all_dataset = {
    'train': {
        'synth': {
            'audio_root': '/home/xingyuan/animal_language2/data/dog_sed_dataset/desed_synth_train_audios',
            'label_path': '/home/xingyuan/animal_language2/data/dog_sed_dataset/desed_synth_train.tsv'
        },
        'audioset': {
            'audio_root': '/home/xingyuan/animal_language2/data/dog_sed_dataset/audioset_strong_audios',
            'label_path': '/home/xingyuan/animal_language2/data/dog_sed_dataset/audioset_strong.tsv'
        },
        'olddataset': {
            'audio_root': '/home/xingyuan/animal_language2/data/dog_sed_dataset/youtube_strong_train_audios_w10_s5',
            'label_path': '/home/xingyuan/animal_language2/data/dog_sed_dataset/youtube_strong_train_w10_s5.tsv'
        },
    },
    'valid': {
        'olddataset1': {
            'audio_root': '/home/xingyuan/animal_language2/data/dog_sed_dataset/youtube_strong_valid_audios',
            'label_path': '/home/xingyuan/animal_language2/data/dog_sed_dataset/youtube_strong_valid.tsv',
            'duration_path': '/home/xingyuan/animal_language2/data/dog_sed_dataset/youtube_strong_valid_duration.tsv'
        },
        'olddataset2': {
            'audio_root': '/home/xingyuan/animal_language2/data/dog_sed_dataset/youtube_strong_valid2_audios',
            'label_path': '/home/xingyuan/animal_language2/data/dog_sed_dataset/youtube_strong_valid2.tsv',
            'duration_path': '/home/xingyuan/animal_language2/data/dog_sed_dataset/youtube_strong_valid2_duration.tsv'
        },
    }
}

def get_sed_duration(path):
    sed_durations = {}
    with open(path, 'r') as file:
        duration_data = [line.strip().split() for line in file.readlines() if len(line.strip()) > 0]
    for dd in duration_data[1:]:
        assert dd[0] not in sed_durations.keys()
        sed_durations[dd[0]] = float(dd[1])
    return sed_durations


def interval_union(clips, interval_time=1e-3):
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
            new_clips.append([start, end, 'dog'])
            start = clips[i][0]
            end = clips[i][1]
    new_clips.append([start, end, 'dog'])
    return new_clips


def get_sed_ground_truth(path, audio_root, test_positive_labels):
    seds_ground_truth = {f:[] for f in os.listdir(audio_root)}
    with open(path, 'r') as file:
        gt_data = [line.strip().split() for line in file.readlines() if len(line.strip()) > 0]
    for gd in gt_data[1:]:
        if gd[3] in test_positive_labels:
            seds_ground_truth[gd[0]].append([float(gd[1]), float(gd[2]), 'dog'])
    for key in seds_ground_truth.keys():
        seds_ground_truth[key] = interval_union(seds_ground_truth[key])
    return seds_ground_truth


def get_olddataset_valid(test_positive_labels):
    global all_dataset
    from dataset2 import DogDataset
    seds_durations = {}
    seds_ground_truth = {}

    dataset1 = DogDataset(audio_root=all_dataset['valid']['olddataset1']['audio_root'],
                          label_path=all_dataset['valid']['olddataset1']['label_path'],
                          args=args,
                          mode='valid')
    datasets = [dataset1]
    # duration
    sed_durations1 = get_sed_duration(all_dataset['valid']['olddataset1']['duration_path'])
    for key in sed_durations1.keys():
        assert key not in seds_durations.keys()
        seds_durations[key] = sed_durations1[key]
    # gt
    sed_ground_truth1 = get_sed_ground_truth(path=all_dataset['valid']['olddataset1']['label_path'],
                                             audio_root=all_dataset['valid']['olddataset1']['audio_root'],
                                             test_positive_labels=test_positive_labels)
    for key in sed_ground_truth1.keys():
        assert key not in seds_ground_truth.keys()
        seds_ground_truth[key] = sed_ground_truth1[key]

    if 'olddataset2' in all_dataset['valid'].keys():
        dataset2 = DogDataset(audio_root=all_dataset['valid']['olddataset2']['audio_root'],
                              label_path=all_dataset['valid']['olddataset2']['label_path'],
                              args=args,
                              mode='valid')
        datasets = [dataset1, dataset2]
        # duration
        sed_durations2 = get_sed_duration(all_dataset['valid']['olddataset2']['duration_path'])
        for key in sed_durations2.keys():
            assert key not in seds_durations.keys()
            seds_durations[key] = sed_durations2[key]
        # gt
        sed_ground_truth2 = get_sed_ground_truth(path=all_dataset['valid']['olddataset2']['label_path'],
                                                 audio_root=all_dataset['valid']['olddataset2']['audio_root'],
                                                 test_positive_labels=test_positive_labels)
        for key in sed_ground_truth2.keys():
            assert key not in seds_ground_truth.keys()
            seds_ground_truth[key] = sed_ground_truth2[key]
    return ConcatDataset(datasets), seds_durations, seds_ground_truth

if __name__ == '__main__':
    args = get_args()
    args.train_positive_labels = args.train_positive_labels.split(',')
    args.test_positive_labels = args.test_positive_labels.split(',')
    set_seed(seed=args.seed, speed_up=False)

    model, frames_per_second = get_model()
    device = torch.device('cpu')

    # 300
    d, seds_durations, seds_ground_truth = get_olddataset_valid(test_positive_labels=args.test_positive_labels)
    dataloader = DataLoader(dataset=d,
                            batch_size=1,
                            shuffle=False,
                            num_workers=2,
                            pin_memory=False)
    test_pred = {}
    for test_batch_dict in tqdm(dataloader):
        test_waveform = test_batch_dict['waveform'].to(torch.float32)
        # test_waveform = torch.tensor([test_batch_dict['waveform']], dtype=torch.float32)
        test_target = test_batch_dict['target'].to(torch.float32)

        with torch.no_grad():
            model.eval()
            test_batch_output = model(test_waveform.to(device), None)
        test_framewise_pred = test_batch_output['framewise_output'][:, :, 0]
        test_id = test_batch_dict['name'][0]
        test_pred[test_id] = test_batch_output['framewise_output'][0, :, 0].cpu().detach().numpy()
    seds_result = get_seds_pred(test_results=test_pred,
                                frames_per_second=frames_per_second,
                                seds_ground_truth0=seds_ground_truth,
                                seds_durations0=seds_durations)


    # 175
    # label_audios = []
    # for file in os.listdir(args.test_labels_root):
    #     id = file[:-len('.TextGrid')]
    #     r = {
    #         'label_path': os.path.join(args.test_labels_root, file),
    #         'audio_path': os.path.join(args.audios_root, f'{id}.wav')
    #     }
    #     label_audios.append(r)
    # test_dataset1 = DogDataset(args=args, mode='test')
    # args.test_labels_root = '/home/xingyuan/animal_language2/data/olddataset2/labels'
    # args.audios_root = '/home/xingyuan/animal_language2/data/olddataset2/audios'
    # test_dataset2 = DogDataset(args=args, mode='test')
    # for file in os.listdir(args.test_labels_root):
    #     id = file[:-len('.TextGrid')]
    #     r = {
    #         'label_path': os.path.join(args.test_labels_root, file),
    #         'audio_path': os.path.join(args.audios_root, f'{id}.wav')
    #     }
    #     label_audios.append(r)
    # test_dataset = ConcatDataset([test_dataset1, test_dataset2])
    #
    # test_dataloader = DataLoader(dataset=test_dataset, batch_size=1,
    #                              shuffle=False, num_workers=2, pin_memory=False)
    # device = torch.device('cpu')
    #
    # test_pred = {}
    # for test_batch_dict in tqdm(test_dataloader):
    #     test_waveform = test_batch_dict['waveform'].to(torch.float32)
    #     with torch.no_grad():
    #         model.eval()
    #         test_batch_output = model(test_waveform.to(device), None)
    #     test_framewise_pred = test_batch_output['framewise_output'][0, :, 0]
    #     test_id = test_batch_dict['name'][0]
    #     test_pred[test_id] = test_framewise_pred.cpu().detach().numpy()
    #
    #
    #
    #
    # seds_result = myeval(label_audios=label_audios,
    #                      test_results=test_pred,
    #                       label_root=args.test_labels_root,
    #                             frames_per_second=frames_per_second,
    #                             audios_root=args.audios_root,
    #                             sr=args.sr,
    #                             test_positive_labels=args.test_positive_labels)
    # with open('./about_f1score.json', 'w+', encoding='utf-8') as file:
    #     json.dump(seds_result, file, indent=4, ensure_ascii=False)

