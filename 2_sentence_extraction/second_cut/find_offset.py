import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import textgrid

sys.path.append('../')
sys.path.append('../../')
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import torch

from pytorch.models import *
from torch.utils.data import DataLoader

from dataset import *
from utils import config

checkpoint_path = '/bigdrive/xingyuan/v2/out/second_cut/panns/alldata_thr0.05_offset1_0-0.01_0.9/[train]_2024-05-30_00-18-15/checkpoints/checkpoint_51_2910.pth'
checkpoint_path = '/home/xingyuan/animal_language2/code/second_cut/audioset_tagging_cnn-master/checkpoints/Cnn14_DecisionLevelMax_mAP=0.385.pth'
device = 1
# offset = 0.
# threshold = 0.2 # 0.2859
device = torch.device(f'cuda:{device}')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, required=False,
                        default=device,
                        help='The number of GPU (-1 is CPU)')
    parser.add_argument('--seed', type=int, required=False,
                        default=42,
                        help='The seed')

    # train
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
    parser.add_argument('--test_labels_root', type=str, required=False,
                        default='/home/xingyuan/animal_language2/data/train_second_cut/alldata_thr0.05_offset1_0-0.01_0.9/valid_labels')

    return parser.parse_args()

def load_my_model2():
    sample_rate = 32000
    window_size = 1024
    hop_size = 320
    mel_bins = 64
    fmin = 50
    fmax = 14000
    model_type = 'Cnn14_DecisionLevelMax'
    classes_num = config.classes_num
    frames_per_second = sample_rate // hop_size

    # Model
    Model = eval(model_type)
    model = Model(sample_rate=sample_rate, window_size=window_size,
                  hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax,
                  classes_num=classes_num)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    return model, frames_per_second, hop_size

def load_my_model():
    sample_rate = 32000
    window_size = 1024
    hop_size = 320
    mel_bins = 64
    fmin = 50
    fmax = 14000
    model_type = 'Cnn14_DecisionLevelMax'
    classes_num = 1
    frames_per_second = sample_rate // hop_size

    # Model
    Model = eval(model_type)
    model = Model(sample_rate=sample_rate, window_size=window_size,
                  hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax,
                  classes_num=classes_num)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    return model, frames_per_second, hop_size

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


def sound_event_detection(offset, threshold, waveform, frames_per_second, duration, start_time=0.0):
    waveform = waveform[None, :]    # (1, audio_length)
    waveform = move_data_to_device(waveform, device)

    # Forward
    with torch.no_grad():
        model.eval()
        batch_output_dict = model(waveform, None)

    framewise_output = batch_output_dict['framewise_output'].data.cpu().numpy()[0]
    """(time_steps, classes_num)"""

    dog_label_index = 0
    dog_framewise_output = framewise_output[:, dog_label_index]
    filtered_dog_framewise_output = np.where(dog_framewise_output >= threshold, 1, 0)
    continue_regions = find_contiguous_regions(filtered_dog_framewise_output)
    continue_regions = continue_regions / frames_per_second

    continue_regions[:, 0] -= offset
    continue_regions[:, 1] += offset
    continue_regions = np.maximum(continue_regions, 0.)
    continue_regions = np.minimum(continue_regions, duration)
    continue_regions += start_time
    clips = continue_regions.tolist()
    clips.sort(key=lambda c: c[0])
    return interval_union(clips=clips, interval_time=1.0)


def sound_event_detection2(offset, threshold, waveform, frames_per_second, duration, start_time=0.0):
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
    filtered_dog_framewise_output = np.where(dog_framewise_output >= threshold, 1, 0)
    continue_regions = find_contiguous_regions(filtered_dog_framewise_output)
    continue_regions = continue_regions / frames_per_second

    continue_regions[:, 0] -= offset
    continue_regions[:, 1] += offset
    continue_regions = np.maximum(continue_regions, 0.)
    continue_regions = np.minimum(continue_regions, duration)
    continue_regions += start_time
    clips = continue_regions.tolist()
    clips.sort(key=lambda c: c[0])
    return interval_union(clips=clips, interval_time=1.0)


def sound_event_detection_soft(offset, threshold, waveform, frames_per_second, duration, start_time=0.0):
    waveform = waveform[None, :]    # (1, audio_length)
    waveform = move_data_to_device(waveform, device)

    # Forward
    with torch.no_grad():
        model.eval()
        batch_output_dict = model(waveform, None)

    framewise_output = batch_output_dict['framewise_output'].data.cpu().numpy()[0]
    """(time_steps, classes_num)"""

    dog_label_index = 0
    dog_framewise_output = framewise_output[:, dog_label_index]

    thr = min(threshold * max(dog_framewise_output), 0.5)
    thr = max(thr, 0.05)

    filtered_dog_framewise_output = np.where(dog_framewise_output >= thr, 1, 0)
    continue_regions = find_contiguous_regions(filtered_dog_framewise_output)
    continue_regions = continue_regions / frames_per_second

    continue_regions[:, 0] -= offset
    continue_regions[:, 1] += offset
    continue_regions = np.maximum(continue_regions, 0.)
    continue_regions = np.minimum(continue_regions, duration)
    continue_regions += start_time
    clips = continue_regions.tolist()
    clips.sort(key=lambda c: c[0])
    return interval_union(clips=clips, interval_time=1.0)

def sound_event_detection_soft2(offset, threshold, waveform, frames_per_second, duration, start_time=0.0):
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

    thr = min(threshold * max(dog_framewise_output), 0.5)
    thr = max(thr, 0.05)

    filtered_dog_framewise_output = np.where(dog_framewise_output >= thr, 1, 0)
    continue_regions = find_contiguous_regions(filtered_dog_framewise_output)
    continue_regions = continue_regions / frames_per_second

    continue_regions[:, 0] -= offset
    continue_regions[:, 1] += offset
    continue_regions = np.maximum(continue_regions, 0.)
    continue_regions = np.minimum(continue_regions, duration)
    continue_regions += start_time
    clips = continue_regions.tolist()
    clips.sort(key=lambda c: c[0])
    return interval_union(clips=clips, interval_time=1.0)


def work_one(offset):
    counts = []
    pred_s_list = []
    for thr in tqdm(np.arange(0.05, 0.51, 0.01)):
        count = 0
        pred_s = 0.
        tmp_count = 0
        for test_batch_dict in test_dataloader:
            wav = test_batch_dict['waveform'][0]
            duration=wav.shape[0] / 32000
            clips = sound_event_detection2(offset=offset,
                                          threshold=thr,
                                          waveform=wav,
                                          frames_per_second=frames_per_second,
                                          duration=duration)

            target = test_batch_dict['target'][0].cpu().detach().numpy().astype(np.int16)
            pred = np.zeros(target.shape[0], dtype=np.int16)
            for clip in clips:
                pred[int(clip[0] * 32000): int(clip[1] * 32000)] = 1
                pred_s += (clip[1] - clip[0])
            pred = pred.astype(np.int16)
            tmp = np.bitwise_and(target, pred)
            tmp_count += np.sum(tmp)
            count += (np.sum(target) - np.sum(tmp))
        count = ((count / 32000) * 1000.) / 322
        counts.append(count)
        pred_s_list.append(((tmp_count / 32000) * 100) / pred_s)
    return np.arange(0.05, 0.51, 0.01), counts, pred_s_list



def work_one_soft(offset):
    counts = []
    pred_s_list = []
    for thr in tqdm(np.arange(0.2, 0.95, 0.05)):
        count = 0
        pred_s = 0.
        tmp_count = 0
        for test_batch_dict in test_dataloader:
            wav = test_batch_dict['waveform'][0]
            duration=wav.shape[0] / 32000
            clips = sound_event_detection_soft2(offset=offset,
                                          threshold=thr,
                                          waveform=wav,
                                          frames_per_second=frames_per_second,
                                          duration=duration)

            target = test_batch_dict['target'][0].cpu().detach().numpy().astype(np.int16)
            pred = np.zeros(target.shape[0], dtype=np.int16)
            for clip in clips:
                pred[int(clip[0] * 32000): int(clip[1] * 32000)] = 1
                pred_s += (clip[1] - clip[0])
            pred = pred.astype(np.int16)
            tmp = np.bitwise_and(target, pred)
            tmp_count += np.sum(tmp)
            count += (np.sum(target) - np.sum(tmp))
        count = ((count / 32000) * 1000.) / 322
        counts.append(count)
        pred_s_list.append(((tmp_count / 32000) * 100) / pred_s)
    return np.arange(0.05, 0.51, 0.01), counts, pred_s_list

"""
human data人声比例：0.8062138857369198
"""
if __name__ == '__main__':
    # args = get_args()
    # args.train_positive_labels = args.train_positive_labels.split(',')
    # args.test_positive_labels = args.test_positive_labels.split(',')
    # model, frames_per_second, hop_size = load_my_model2()
    # test_dataset = DogDataset(args=args, mode='test')
    # test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=False)

    # # 80,55: 0-19-15.883796414931638-39.80046583850932-55.684262253440956
    # # 80,60: 0-33--6.581293061489461-66.7542701863354-60.17297712484594
    # # 100,60:0-33--6.581293061489461-66.7542701863354-60.17297712484594
    # # 100,65:-1
    # # 100,55: 0-19-15.883796414931638-39.80046583850932-55.684262253440956
    # with open('./loss_hard1.0.txt', 'r') as file:
    #     losses = [line.strip().split(',') for line in file.readlines() if len(line.strip()) > 0]
    # with open('./value_hard1.0.txt', 'r') as file:
    #     values = [line.strip().split(',') for line in file.readlines() if len(line.strip()) > 0]

    # # 80,55: 0-8-20.31452149596494-35.103163819875775-55.417685315840714
    # # 80,60: 0-23--4.247593661500858-64.39081909937889-60.14322543787803
    # # 100,60:0-23--4.247593661500858-64.39081909937889-60.14322543787803
    # # 100,65:-1
    # # 100,55: 0-8-20.31452149596494-35.103163819875775-55.417685315840714
    # with open('./loss_hard0.5.txt', 'r') as file:
    #     losses = [line.strip().split(',') for line in file.readlines() if len(line.strip()) > 0]
    # with open('./value_hard0.5.txt', 'r') as file:
    #     values = [line.strip().split(',') for line in file.readlines() if len(line.strip()) > 0]

    # # 80,55: 1-4-9.254203969619809-46.17760093167702-55.43180490129683
    # # 80,60: -1
    # # 100,60: 0-7--33.61293512208281-94.2462150621118-60.63327994002899
    # # 100,65: -1
    # # 100,55: 1-4-9.254203969619809-46.17760093167702-55.43180490129683
    # with open('./loss_soft1.0.txt', 'r') as file:
    #     losses = [line.strip().split(',') for line in file.readlines() if len(line.strip()) > 0]
    # with open('./value_soft1.0.txt', 'r') as file:
    #     values = [line.strip().split(',') for line in file.readlines() if len(line.strip()) > 0]

    # # 80,55,: 0-0-16.29687705619059-39.691187888198755-55.98806494438934
    # # 80,60 : 0-4--17.562419129000588-78.3288043478261-60.766385218825505
    # # 100,60: 0-4--17.562419129000588-78.3288043478261-60.766385218825505
    # # 100,65: -1
    # # 100,55: 0-0-16.29687705619059-39.691187888198755-55.98806494438934
    # with open('./loss_soft0.5.txt', 'r') as file:
    #     losses = [line.strip().split(',') for line in file.readlines() if len(line.strip()) > 0]
    # with open('./value_soft0.5.txt', 'r') as file:
    #     values = [line.strip().split(',') for line in file.readlines() if len(line.strip()) > 0]

    # ---------------------------------

    # 80,55: 0-19-15.883796414931638-39.80046583850932-55.684262253440956
    # 80,60: 0-33--6.581293061489461-66.7542701863354-60.17297712484594
    # 100,60:0-33--6.581293061489461-66.7542701863354-60.17297712484594
    # 100,65:-1
    # 100,55: -1
    with open('./loss2_hard1.0.txt', 'r') as file:
        losses = [line.strip().split(',') for line in file.readlines() if len(line.strip()) > 0]
    with open('./value2_hard1.0.txt', 'r') as file:
        values = [line.strip().split(',') for line in file.readlines() if len(line.strip()) > 0]

    # 80,55: 0-8-20.31452149596494-35.103163819875775-55.417685315840714
    # 80,60: 0-23--4.247593661500858-64.39081909937889-60.14322543787803
    # 100,60:0-23--4.247593661500858-64.39081909937889-60.14322543787803
    # 100,65:-1
    # 100,55: -1
    with open('./loss2_hard0.5.txt', 'r') as file:
        losses = [line.strip().split(',') for line in file.readlines() if len(line.strip()) > 0]
    with open('./value2_hard0.5.txt', 'r') as file:
        values = [line.strip().split(',') for line in file.readlines() if len(line.strip()) > 0]

    # 80,55: 1-4-9.254203969619809-46.17760093167702-55.43180490129683
    # 80,60: -1
    # 100,60: 0-7--33.61293512208281-94.2462150621118-60.63327994002899
    # 100,65: -1
    # 100,55: -1
    with open('./loss2_soft1.0.txt', 'r') as file:
        losses = [line.strip().split(',') for line in file.readlines() if len(line.strip()) > 0]
    with open('./value2_soft1.0.txt', 'r') as file:
        values = [line.strip().split(',') for line in file.readlines() if len(line.strip()) > 0]

    # 80,55,: 0-0-16.29687705619059-39.691187888198755-55.98806494438934
    # 80,60 : 0-4--17.562419129000588-78.3288043478261-60.766385218825505
    # 100,60: 0-4--17.562419129000588-78.3288043478261-60.766385218825505
    # 100,65: -1
    # 100,55: -1
    with open('./loss2_soft0.5.txt', 'r') as file:
        losses = [line.strip().split(',') for line in file.readlines() if len(line.strip()) > 0]
    with open('./value2_soft0.5.txt', 'r') as file:
        values = [line.strip().split(',') for line in file.readlines() if len(line.strip()) > 0]


    mq = -100.
    mloss = 100000.
    mvalue = -10000.
    mi = -1
    mj = -1
    for i in range(len(losses)):
        for j in range(len(losses[i])):
            loss = float(losses[i][j])
            value = float(values[i][j])
            if loss >= 100.: continue
            if value <= 55.: continue
            q = value - loss
            flag = False
            if q > mq:
                flag = True
            elif mq == q:
                if loss < mloss:
                    flag = True

            if flag:
                mq = q
                mi = i
                mj = j
                mloss = loss
                mvalue = value
    print(f'{mi}-{mj}-{mq}-{mloss}-{mvalue}')


    # colors = [
    #     '#FF0000', '#FF8000', '#FFFF00', '#80FF00', '#0000FF',
    #     '#8000FF', '#FF00FF', '#FF3D3D', '#00FFFF', 'black'
    # ]
    # ind = 0
    # for offset in np.arange(0.00, 1.0, 0.1):
    #     x, y, z = work_one(offset)
    #     with open('./loss2_hard1.0.txt', 'a+') as file:
    #         file.write(','.join([str(d) for d in y]))
    #         file.write('\n')
    #     with open('./value2_hard1.0.txt', 'a+') as file:
    #         file.write(','.join([str(d) for d in z]))
    #         file.write('\n')
    #
    # for offset in np.arange(0.00, 1.0, 0.1):
    #     x, y, z = work_one_soft(offset)
    #     with open('./loss2_soft1.0.txt', 'a+') as file:
    #         file.write(','.join([str(d) for d in y]))
    #         file.write('\n')
    #     with open('./value2_soft1.0.txt', 'a+') as file:
    #         file.write(','.join([str(d) for d in z]))
    #         file.write('\n')


    # colors = [
    #     '#FF0000', '#FF8000', '#FFFF00', '#80FF00', '#0000FF',
    #     '#8000FF', '#FF00FF', '#FF3D3D', '#00FFFF', 'black'
    # ]
    # ind = 0
    # for offset in np.arange(0.00, 1.0, 0.1):
    #     x, y = work_one(offset)
    #     plt.plot(x, y, c=colors[ind], label=str(offset), alpha=0.5)
    #     ind += 1
    # plt.legend()
    # plt.xlabel('thr')
    # plt.savefig("0.00-1.0-0.1_1.0.png")


    # # target mean: 1.0318947204968942
    # # 0.3, 0.18, 0.25: [1.2771142362794607, 1.2771142362794607, 1.2771142362794607, 1.2831334357199162, 1.32972580138869, 1.4260329924359734, 1.4260329924359734, 1.5415922166935878, 1.5415922166935878, 1.6935958125633337, 1.789903003610617, 2.0123124229354374, 2.0123124229354374, 2.0123124229354374, 2.578803735274405, 2.578803735274405, 2.578803735274405, 2.5894501942847103, 2.5894501942847103, 2.5894501942847103, 2.602240993095678, 2.6233552161328997, 2.7188347672571207, 2.7797603515934783, 2.7857795510339334]
    # # 0.4, 0.18, 0.25: [1.1434033637092234, 1.1434033637092234, 1.1434033637092234, 1.1494601831461817, 1.1960525488149552, 1.2923597398622388, 1.2923597398622388, 1.3778135619184515, 1.3778135619184515, 1.5298171577881972, 1.6032984159573795, 1.8257078352822, 1.8257078352822, 1.8257078352822, 2.1256708773976354, 2.1256708773976354, 2.1256708773976354, 2.1256708773976354, 2.1256708773976354, 2.1256708773976354, 2.1256708773976354, 2.1768810976371333, 2.2422646515590783, 2.27309423869316, 2.27309423869316]
    # offset = 0.4
    # x, y = work_one(offset)
    # plt.plot(x, y, c='red', label=str(offset), alpha=1.0)
    # plt.legend()
    # plt.xlabel('thr')
    # plt.savefig(f"{offset}_1.0.png")
    # print(y)

    # ##################### 0.3-0.18: 3.8545830965909094
    # # 0.3-0.25: 3.78503851744186
    # # 0.4-0.25: 4.092690512048192
    # # 0.4-0.18: 4.097596656976744
    # pred_count = 0
    # threshold = 0.18
    # offset = 0.4
    # total_clips = []
    # for test_batch_dict in tqdm(test_dataloader):
    #     wav = test_batch_dict['waveform'][0]
    #     duration=wav.shape[0] / 32000
    #     clips = sound_event_detection(offset=offset,
    #                                   threshold=threshold,
    #                                   waveform=wav,
    #                                   frames_per_second=frames_per_second,
    #                                   duration=duration)
    #
    #     # target = test_batch_dict['target'][0].cpu().detach().numpy().astype(np.int16)
    #     # pred = np.zeros(target.shape[0], dtype=np.int16)
    #     # for clip in clips:
    #     #     pred[int(clip[0] * 32000): int(clip[1] * 32000)] = 1
    #     # pred = pred.astype(np.int16)
    #     # tmp = np.bitwise_and(target, pred)
    #     # pred_count += np.sum(pred)
    #     for clip in clips:
    #         total_clips.append(clip[1] - clip[0])
    # print(len(total_clips))
    # print(min(total_clips))
    # print(max(total_clips))
    # print(np.mean(total_clips))


    # clip_count = get_target_count()
    #
    # print(f'm: {count / 32000}')
    # print(f'target: {target_count / 32000}')
    # print(f'm / target: {count / target_count}')
    # print(f'target clip count: {clip_count}')
    # print(f'm / target clip count: {(count / 32000) / clip_count}')


