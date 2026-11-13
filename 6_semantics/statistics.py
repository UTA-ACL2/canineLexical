import bisect
import copy
import json
import os
import shutil

import librosa
import matplotlib.pyplot as plt
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
from tqdm import tqdm


def merge_result():
    root = '/home2/xingyuan/animal_language/v2/out/context/mmaction'
    save_path = './train_activity.jsonlines'
    files = []
    for file in os.listdir(root):
        id = int(file.split('.')[0][len('train_'):])
        files.append([os.path.join(root, file), id])
    files.sort(key=lambda f: f[1])

    results = []
    for file in tqdm(files):
        with open(file[0], 'r') as f:
            results.extend(f.readlines())
    with open(save_path, 'w+') as file:
        file.write(''.join(results))
    print(len(results))

{'name': '108-113-59-83-17-77', 'start_time': 3464.16, 'end_time': 3464.41}
{'name': '[UCeqTOU_4b6UEmPgBdqd4YWg]_[eZa4OoV216g]_00141_00000', 'video_path': '/home2/xingyuan/animal_language/data/videos/dog/Chihuahua/[UCeqTOU_4b6UEmPgBdqd4YWg]_[eZa4OoV216g].mp4', 'audio_path': '/home2/xingyuan/animal_language/data/audios/dog/Chihuahua/[UCeqTOU_4b6UEmPgBdqd4YWg]_[eZa4OoV216g].wav', 'start': 3463.96, 'end': 3464.6, 'clips': [{'start': 3458.96, 'end': 3469.66, 'class_id': 10, 'class_label': 'Playing with human'}]}

def analysis(word_info, action_info):
    result = {
        'left': [],
        'left_overlap': [],
        'partial_overlap': [],
        'all_overlap': [],
        'right_overlap': [],
        'right': []
    }

    word_start = word_info['start_time']
    word_end = word_info['end_time']
    for action_clip in action_info['clips']:
        clip_start = action_clip['start']
        clip_end = action_clip['end']
        if clip_end <= word_start:
            result['left'].append(action_clip)
            continue
        if (clip_start < word_start < clip_end) and (word_start < clip_end < word_end):
            result['left_overlap'].append(action_clip)
            continue
        if word_start <= clip_start < clip_end <= word_end and ((word_start < clip_start) or (clip_end < word_end)):
            result['partial_overlap'].append(action_clip)
            continue
        if clip_start <= word_start < word_end <= clip_end:
            result['all_overlap'].append(action_clip)
            continue
        if (word_start < clip_start < word_end) and (clip_start < word_end < clip_end):
            result['right_overlap'].append(action_clip)
            continue
        if word_end <= clip_start:
            result['right'].append(action_clip)
            continue

    return result


def analysis2(word_info, action_info, thr1=0.5, thr2=0.3):
    result = {
        'left': [],
        'overlap': [],
        'right': []
    }

    word_start = word_info['start_time']
    word_end = word_info['end_time']
    for action_clip in action_info['clips']:
        clip_start = action_clip['start']
        clip_end = action_clip['end']
        if clip_end <= word_start:
            result['left'].append(action_clip)
            continue
        if word_end <= clip_start:
            result['right'].append(action_clip)
            continue
        if word_start <= clip_start < clip_end <= word_end:
            result['overlap'].append(action_clip)
            continue

        if word_start < clip_end <= word_end:
            if ((word_start - clip_start) / (clip_end - clip_start)) >= thr1:
                result['left'].append(action_clip)
            else:
                result['overlap'].append(action_clip)
            continue

        if word_start <= clip_start < word_end:
            if ((clip_end - word_end) / (clip_end - clip_start)) >= thr1:
                result['right'].append(action_clip)
            else:
                result['overlap'].append(action_clip)
            continue

        if clip_start <= word_start < word_end <= clip_end:
            assert (clip_start < word_start) or (word_end < clip_end)
            left = word_start - clip_start
            right = clip_end - word_end
            overlap = word_end - word_start
            flag_left = (left / (clip_end - clip_start)) >= thr2
            flag_right = (right / (clip_end - clip_start)) >= thr2
            if (overlap >= left) and (overlap >= right):
                result['overlap'].append(action_clip)
                continue
            if flag_left and flag_right:
                result['left'].append(action_clip)
                result['overlap'].append(action_clip)
                result['right'].append(action_clip)
                continue
            if flag_left and (not flag_right):
                result['left'].append(action_clip)
                continue
            if flag_right and (not flag_left):
                result['right'].append(action_clip)
                continue

    return result


# root = '/home2/xingyuan/animal_language/v2/out/remove_noise/audioset_thr-0.7'
# audioset_files = [f for f in os.listdir(root)]
# audioset_files.sort()
def get_user(name):
    global audioset_files
    ind = bisect.bisect_left(audioset_files, name)
    head = '_'.join(name.split('_')[0: -2])
    if (0 <= ind < len(audioset_files)) and (audioset_files[ind] == name):
        return head
    return '_'.join(head.split('_')[:-1])

def generate_p(word2result, key, all_count, user_count, word2users):
    probability = [['word', 'Standing', 'Walking', 'Sitting', 'Laying down', 'Eating', 'Sleeping', 'Running', 'No dog', 'Taking a shower', 'Sniffing', 'Playing with human', 'Playing with a toy', 'Playing with dogs', 'Playing with other animals', 'Swimming', 'begging for food']]
    for word in tqdm(word2result.keys()):
        counts = [0 for _ in range(16)]
        counts.insert(0, word)
        for result in word2result[word]:
            for clip in result[key]:
                index = clip['class_id'] + 1
                assert probability[0][index] == clip['class_label']
                counts[index] += 1
        probability.append(counts)

    probability[0].append('total')
    for p in probability[1:]:
        p.append(sum(p[1:]))

    tfs = [p[-1] / all_count for p in probability[1:]]
    idfs = [len(word2users[p[0]]) / user_count for p in probability[1:]]
    tf_idfs = [tfs[i] * idfs[i] for i in range(len(tfs))]
    probability[0].append('tf-idf')
    for i in range(len(probability[1:])):
        probability[i + 1].append(tf_idfs[i])

    total = [0 for _ in range(len(probability[0]) - 1)]
    total.insert(0, 'sum')
    for i in range(1, len(probability[0])):
        total[i] = sum([p[i] for p in probability[1:]])
    probability.append(total)
    with open(f'./c_{key}.csv', 'w+') as file:
        for p in probability:
            file.write(','.join([str(l) for l in p]))
            file.write('\n')

    probability_by_word = copy.deepcopy(probability)
    for pr in probability_by_word[1:]:
        if pr[-2] > 0:
            for i in range(1, len(pr) - 2):
                pr[i] /= pr[-2]
    with open(f'./c_{key}_by_word.csv', 'w+') as file:
        for p in probability_by_word:
            file.write(','.join([str(l) for l in p]))
            file.write('\n')

    probability_by_action = copy.deepcopy(probability)
    for pr in probability_by_action[1: -1]:
        for i in range(1, len(pr) - 2):
            if probability_by_action[-1][i] > 0:
                pr[i] /= probability_by_action[-1][i]
    with open(f'./c_{key}_by_action.csv', 'w+') as file:
        for p in probability_by_action:
            file.write(','.join([str(l) for l in p]))
            file.write('\n')


def statistic():
    dictionary_path = '/home2/xingyuan/animal_language/v2/code/context/mmaction/work/energy_vol/dictionary_log.csv'
    action_result_path = '/home2/xingyuan/animal_language/v2/out/context/mmaction/train_activity.jsonlines'

    dictionary_path = '/home/sinong/ALP/DogRecognitionLabeling/grammar_induction/PyAdaGram/short-dog/words/words_info/24Jun14-215338-D31294-P10-S10-B128-O244-t512-k0.8-Ggrammar.unigram.csv'
    action_result_path = '/home/xingyuan/animal_language2/work/train_activity.jsonlines'


    key2action = {}
    with open(action_result_path, 'r') as file:
        audios = [json.loads(line.strip()) for line in file.readlines() if len(line.strip()) > 0]
    for audio in tqdm(audios):
        key = f'{audio["name"]}.wav'
        assert key not in key2action.keys(), key
        key2action[key] = audio

    word2result = {}
    result_keys = []
    users = []
    word2users = {}
    with open(dictionary_path, 'r') as file:
        lines = [line.strip() for line in file.readlines() if len(line.strip()) > 0]
    for line in tqdm(lines[1:]):
        infos = line.split(',')
        key = infos[1]
        word = infos[0]
        word_info = {
            'name': word,
            'start_time': float(infos[3]),
            'end_time': float(infos[4])
        }
        user = get_user(key)
        users.append(user)
        if key in key2action.keys():
            action_info = key2action[key]
            offset = action_info['start']
            word_info['start_time'] += offset
            word_info['end_time'] += offset
            result = analysis2(word_info=word_info, action_info=action_info)
            result_keys = [k for k in result.keys()]
            result['word'] = word_info['name']
            result['sentence_name'] = action_info['name']
            if word not in word2result.keys():
                word2result[word] = [result]
            else:
                word2result[word].append(result)

            if word not in word2users.keys():
                word2users[word] = [user]
            else:
                word2users[word].append(user)

    users = list(set(users))
    for word in word2users.keys():
        word2users[word] = list(set(word2users[word]))
    for k in result_keys:
        generate_p(word2result,
                   key=k,
                   all_count=len(lines) - 1,
                   user_count=len(users),
                   word2users=word2users)


def get_top100(input_path, save_root, remove_noise=True, remove_one=False):
    with open(input_path, 'r') as file:
        lines = [line.strip().split(',') for line in file.readlines() if len(line.strip()) > 0]
    head = copy.deepcopy(lines[0])
    tail = copy.deepcopy(lines[-1])
    lines.pop(0)
    lines.pop(-1)
    topk = 200
    phoneme_min_count = 3

    # 移除noise超过一半的word、total小于15的word
    noise_labels = [4, 10, 20, 23, 24, 32, 37, 39, 50, 52, 53, 54, 65, 71, 72, 78, 79, 82, 85, 87, 89, 108, 109, 114, 119, 120, 122, 134, 137]
    if remove_noise:
        remove_index = []
        for i in range(len(lines)):
            labels = [int(l) for l in lines[i][0].split('-')]
            count = 0
            for l in labels:
                if l in noise_labels:
                    count += 1
            if count / len(labels) >= 0.5 or (int(lines[i][-2]) < 15):
                remove_index.append(i)

            # flag = False
            # for l in labels:
            #     if l not in noise_labels:
            #         flag = True
            #         break
            # if not flag:
            #     remove_index.append(i)
        remove_index.sort(key=lambda ri: -ri)
        for ri in remove_index:
            lines.pop(ri)



    lines.sort(key=lambda l: -float(l[-2]))
    lines1 = lines[0: topk]
    # if remove_one:
    #     remove_index = []
    #     for i in range(len(lines1)):
    #         if len(lines1[i][0].split('-')) <= phoneme_min_count:
    #             remove_index.append(i)
    #     remove_index.sort(key=lambda ri: -ri)
    #     for ri in remove_index:
    #         lines1.pop(ri)
    with open(os.path.join(save_root, f'{os.path.basename(input_path)[:-len(".csv")]}_total.csv'), 'w+') as file:
        file.write(','.join(head))
        file.write('\n')
        for line in lines1:
            file.write(','.join([str(l) for l in line]))
            file.write('\n')
        file.write(','.join(tail))
        file.write('\n')

    lines.sort(key=lambda l: -float(l[-1]))
    lines2 = lines[0: topk]
    # if remove_one:
    #     remove_index = []
    #     for i in range(len(lines2)):
    #         if len(lines2[i][0].split('-')) <= phoneme_min_count:
    #             remove_index.append(i)
    #     remove_index.sort(key=lambda ri: -ri)
    #     for ri in remove_index:
    #         lines2.pop(ri)
    with open(os.path.join(save_root, f'{os.path.basename(input_path)[:-len(".csv")]}_tf-idf.csv'), 'w+') as file:
        file.write(','.join(head))
        file.write('\n')
        for line in lines2:
            file.write(','.join([str(l) for l in line]))
            file.write('\n')
        file.write(','.join(tail))
        file.write('\n')


def show_top100():
    root = '/home2/xingyuan/animal_language/v2/code/context/mmaction/work/top100'
    save_root = '/home2/xingyuan/animal_language/v2/code/context/mmaction/work/show_top100'
    all_labels = ['Standing', 'Walking', 'Sitting', 'Laying down', 'Eating', 'Sleeping', 'Running', 'No dog',
                    'Taking a shower', 'Sniffing', 'Playing with human', 'Playing with a toy', 'Playing with dogs',
                    'Playing with other animals', 'Swimming', 'begging for food']

    if os.path.exists(save_root):
        shutil.rmtree(save_root)
    os.makedirs(save_root)

    files = [f for f in os.listdir(root)]
    t_left = [1209,912,401,1989,228,0,276,1519,42,190,1351,505,0,0,7,87]
    t_left = np.array(t_left)
    t_left = t_left / np.sum(t_left)
    t_overlap = [402,334,147,746,74,0,127,553,16,49,612,131,0,0,7,56]
    t_overlap = np.array(t_overlap)
    t_overlap = t_overlap / np.sum(t_overlap)
    t_right = [1669,1027,519,1898,340,0,328,1634,106,315,1642,626,0,0,7,127]
    t_right = np.array(t_right)
    t_right = t_right / np.sum(t_right)
    for file in tqdm(files):
        if file.find('_action_') >= 0:
            continue
        with open(os.path.join(root, file), 'r') as f:
            lines = [line.strip().split(',') for line in f.readlines() if len(line.strip()) > 0]
        for line in lines[1: -1]:
            save_path = os.path.join(save_root, file, f'{line[0]}.png')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            plt.figure(dpi=150)
            y = np.array([float(l) for l in line[1: -2]])
            # plt.plot([i for i in range(len(y))], y, c='blue')
            if file.find('_left_') >= 0:
                t = t_left
            if file.find('_overlap_') >= 0:
                t = t_overlap
            if file.find('_right_') >= 0:
                t = t_right
            plt.plot([i for i in range(len(y))], y - t, c='b')
            plt.plot([0., len(all_labels) - 1], [0., 0.], linestyle='--', c='r')
            plt.xticks([i for i in range(len(y))], labels=all_labels, rotation=80)
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()

v1 = {"name": "OlivertheBeagle_video134_00002_00000", "video_path": "/home/chunhao/AnimalLanguage/data/shibavideos_soft/OlivertheBeagle/video134.mp4", "audio_path": "/home2/xingyuan/animal_language/other_data/ShibaInu_audios/OlivertheBeagle/video134.wav", "start": 30.68, "end": 35.48, "clips": [{"start": 25.68, "end": 29.68, "class_id": 0, "class_label": "Standing"}, {"start": 29.68, "end": 31.68, "class_id": 11, "class_label": "Playing with a toy"}, {"start": 31.68, "end": 35.68, "class_id": 0, "class_label": "Standing"}, {"start": 35.68, "end": 36.624277610944276, "class_id": 11, "class_label": "Playing with a toy"}, {"start": 36.657644310977645, "end": 40.528181514848185, "class_id": 0, "class_label": "Standing"}]}
v2 = {"name": "[UC_RgS4Yj5G9HqR9W2aeLM_A]_[ykyUy1Bk03E]_00000_00001", "video_path": "/home2/xingyuan/animal_language/data/videos/dog/Chihuahua/[UC_RgS4Yj5G9HqR9W2aeLM_A]_[ykyUy1Bk03E].mp4", "audio_path": "/home2/xingyuan/animal_language/data/audios/dog/Chihuahua/[UC_RgS4Yj5G9HqR9W2aeLM_A]_[ykyUy1Bk03E].wav", "start": 6.04, "end": 12.12, "clips": [{"start": 1.04, "end": 13.739999999999998, "class_id": 3, "class_label": "Laying down"}, {"start": 13.739999999999998, "end": 17.14, "class_id": 7, "class_label": "No dog"}]}
v3 = {"name": "shirosuki_video1966_00017_00000", "video_path": "/home/chunhao/AnimalLanguage/data/shibavideos_soft/shirosuki/video1966.mp4", "audio_path": "/home2/xingyuan/animal_language/other_data/ShibaInu_audios/shirosuki/video1966.wav", "start": 226.52, "end": 229.08, "clips": [{"start": 221.52, "end": 229.52, "class_id": 3, "class_label": "Laying down"}, {"start": 229.52, "end": 231.52, "class_id": 11, "class_label": "Playing with a toy"}, {"start": 231.52, "end": 233.52, "class_id": 3, "class_label": "Laying down"}, {"start": 233.52, "end": 234.12, "class_id": 11, "class_label": "Playing with a toy"}]}
v4 = {"name": "ShibaInuKohachannel_video1662_00000_00000", "video_path": "/home/chunhao/AnimalLanguage/data/shibavideos_soft/ShibaInuKohachannel/video1662.mp4", "audio_path": "/home2/xingyuan/animal_language/other_data/ShibaInu_audios/ShibaInuKohachannel/video1662.wav", "start": 0.0, "end": 3.2, "clips": [{"start": 0.0, "end": 4.517241379310345, "class_id": 15, "class_label": "begging for food"}, {"start": 4.551724137931035, "end": 8.206896551724139, "class_id": 7, "class_label": "No dog"}]}
v5 = {"name": "[UCeWOLN5kMB8Qq-Vua2Pa4dA]_[m2W0RzJoiRg]_00004_00005", "video_path": "/home2/xingyuan/animal_language/data/videos/dog/Chihuahua/[UCeWOLN5kMB8Qq-Vua2Pa4dA]_[m2W0RzJoiRg].mp4", "audio_path": "/home2/xingyuan/animal_language/data/audios/dog/Chihuahua/[UCeWOLN5kMB8Qq-Vua2Pa4dA]_[m2W0RzJoiRg].wav", "start": 74.52, "end": 80.28, "clips": [{"start": 69.52, "end": 73.52, "class_id": 7, "class_label": "No dog"}, {"start": 73.52, "end": 81.52, "class_id": 3, "class_label": "Laying down"}, {"start": 81.52, "end": 85.30243333333333, "class_id": 0, "class_label": "Standing"}]}
v6 = {"name": "jiyvHLGcls4_00000_00001", "video_path": "/home2/xingyuan/animal_language/audioset_videos/jiyvHLGcls4.mp4", "audio_path": "/home2/xingyuan/animal_language/audioset_dog/audio_32k/jiyvHLGcls4.npy", "start": 3.84, "end": 8.0, "clips": [{"start": 0.0, "end": 10.0, "class_id": 3, "class_label": "Laying down"}]}



if __name__ == '__main__':
    dictionary_path = '/home/sinong/ALP/DogRecognitionLabeling/grammar_induction/PyAdaGram/short-dog/words/words_info/24Jun14-215338-D31294-P10-S10-B128-O244-t512-k0.8-Ggrammar.unigram.csv'
    action_result_path = '/home/xingyuan/animal_language2/work/train_activity.jsonlines'

    with open(dictionary_path, 'r') as file:
        lines = [line.strip() for line in file.readlines() if len(line.strip()) > 0]
    print(len(lines))

    print(os.path.getsize(dictionary_path))
    # 651931
    # 104238753

    # root = '/home2/xingyuan/animal_language/v2/code/context/mmaction/work/top100'
    # words = []
    # for file in os.listdir(root):
    #     with open(os.path.join(root, file), 'r') as file:
    #         for line in file.readlines():
    #             line = line.strip().split(',')
    #             if len(line) > 0:
    #                 words.append(line[0])
    # words = list(set(words))
    # print(len(words))
    # print(words)
    #
    # path = '/home2/xingyuan/animal_language/v2/code/context/mmaction/work/words_info/24Jun14-215338-D31294-P10-S10-B128-O244-t512-k0.8-Ggrammar.unigram.csv'
    # total = []
    # ok = []
    # with open(path, 'r') as file:
    #     for line in file.readlines():
    #         line = line.strip().split(',')
    #         if len(line) > 0:
    #             total.append(line[1])
    #             if line[0] in words:
    #                 ok.append(line[1])
    # total = list(set(total))
    # ok = list(set(ok))
    # print(len(ok))
    # print(len(total))
    # print(ok)
    # print(total)
    # print(len(ok) / len(total))





    # action_result_path = '/home2/xingyuan/animal_language/v2/out/context/mmaction/train_activity.jsonlines'
    # word_infos = '/home2/xingyuan/animal_language/v2/code/context/mmaction/work/words_info/24Jun14-215338-D31294-P10-S10-B128-O244-t512-k0.8-Ggrammar.unigram.csv'
    #
    # root = '/home2/xingyuan/animal_language/v2/code/context/mmaction/work/phones'
    # files = [f for f in os.listdir(root)]
    # frame_length = 256
    # for f in files:
    #     print(f)
    #     plt.figure(150)
    #     wav, sr = librosa.load(path=os.path.join(root, f), mono=True)
    #     #plt.specgram(wav, Fs=sr, scale='dB', NFFT=frame_length, window=np.hanning(M=frame_length))
    #     plt.specgram(wav, Fs=sr, scale='dB')
    #     # plt.title(f[:-len('.wav')].split('_')[-1])
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(root, f'{f[:-len(".wav")]}.png'))
    #     plt.close()

    # # 切视频
    # v = v6
    # video = VideoFileClip(v['video_path'])
    # v['clips'].sort(key=lambda c: c['start'])
    # start = v['clips'][0]['start']
    # end = v['clips'][-1]['end']
    # sub_video = video.subclip(start, end)
    # sub_video.write_videofile(f'./{v["name"]}.mp4')

    # # 找到sinong要的视频
    # word_key = ['9-16', '43-25-86', '46-3', '59-43-25', '59-74', '92-61', '113-59-83', '113-59', '113-59', '9-16', '12-61', '46-3', '51-22', '59-74', '59-83', '93-51']
    # word_key = list(set(word_key))
    # sentence2wordlist = {}
    # with open(word_infos, 'r') as file:
    #     for line in file.readlines():
    #         line = line.strip().split(',')
    #         if len(line) < 1: continue
    #         if line[0] in word_key:
    #             if line[1] in sentence2wordlist.keys():
    #                 sentence2wordlist[line[1]].append(line[0])
    #             else:
    #                 sentence2wordlist[line[1]] = [line[0]]
    #
    # sentences = [[key, sentence2wordlist[key]] for key in sentence2wordlist.keys()]
    # sentences.sort(key=lambda s: -len(s[1]))
    # for s in sentences:
    #     if len(s[1]) > 1:
    #         print(s)



    # show_top100()

    # # 得到top100
    # input_root = '/home2/xingyuan/animal_language/v2/code/context/mmaction/work'
    # save_root = '/home2/xingyuan/animal_language/v2/code/context/mmaction/work/top100'
    # if os.path.exists(save_root):
    #     shutil.rmtree(save_root)
    # os.makedirs(save_root)
    # input_files = [
    #     'c_left_by_word.csv',
    #     'c_left_by_action.csv',
    #     'c_overlap_by_word.csv',
    #     'c_overlap_by_action.csv',
    #     'c_right_by_word.csv',
    #     'c_right_by_action.csv',
    # ]
    # for file in tqdm(input_files):
    #     get_top100(input_path=os.path.join(input_root, file),
    #                save_root=save_root,
    #                remove_one=True)

    # show_top100()
