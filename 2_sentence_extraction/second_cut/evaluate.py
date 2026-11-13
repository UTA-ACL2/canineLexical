import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dataset import *
from sed_scores_eval import *

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

def get_seds_pred(*, test_results, label_root, frames_per_second, audios_root, sr, test_positive_labels):
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
    seds_ground_truth, seds_durations = get_seds_ground_truth(label_root=label_root, audios_root=audios_root, sr=sr, test_positive_labels=test_positive_labels)

    # ---------------------------- segment based start ----------------------------

    # segment based best error rate
    best_error_rate, _, __, best_threshold, ___ = segment_based.best_error_rate(
        scores=seds_pred,
        ground_truth=seds_ground_truth,
        audio_durations=seds_durations,
        segment_length=segment_length,
        num_jobs=num_jobs
    )
    results['segment_based']['best_error_rate'] = best_error_rate['dog']
    results['segment_based']['best_er_threshold'] = best_threshold['dog']

    # segment based 0.5 error rate
    error_rate, _, __, ___ = segment_based.error_rate(
        scores=seds_pred,
        ground_truth=seds_ground_truth,
        audio_durations=seds_durations,
        threshold=0.5,
        segment_length=segment_length,
        num_jobs=num_jobs
    )
    results['segment_based']['error_rate50'] = error_rate['dog']

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

    # segment based 0.5 f1
    f1, precision, recall, ___ = segment_based.fscore(
        scores=seds_pred,
        ground_truth=seds_ground_truth,
        audio_durations=seds_durations,
        threshold=0.5,
        segment_length=segment_length,
        num_jobs=num_jobs
    )
    results['segment_based']['f1_score50'] = f1['dog']
    results['segment_based']['precision50'] = precision['dog']
    results['segment_based']['recall50'] = recall['dog']

    # segment based best recall
    _, recalls, scores, __ = segment_based.precision_recall_curve(
        scores=seds_pred,
        ground_truth=seds_ground_truth,
        audio_durations=seds_durations,
        segment_length=segment_length,
        num_jobs=num_jobs
    )['dog']
    max_recall = max(recalls)
    max_recall_thr = -1.0
    for i in range(len(recalls)):
        if (max_recall - recalls[i]) < 1e-9:
            max_recall_thr = max(max_recall_thr, scores[i])
    results['segment_based']['best_recall'] = max_recall
    results['segment_based']['best_recall_threshold'] = max_recall_thr

    # ---------------------------- segment based end ----------------------------

    # ---------------------------- collar based start ----------------------------
    # collar based best error rate
    best_error_rate, _, __, best_threshold, ___ = collar_based.best_error_rate(
        scores=seds_pred,
        ground_truth=seds_ground_truth,
        onset_collar=collar,
        offset_collar=collar,
        num_jobs=num_jobs
    )
    results['collar_based']['best_error_rate'] = best_error_rate['dog']
    results['collar_based']['best_er_threshold'] = best_threshold['dog']

    # collar based 0.5 error rate
    error_rate, _, __, ___ = collar_based.error_rate(
        scores=seds_pred,
        ground_truth=seds_ground_truth,
        onset_collar=collar,
        offset_collar=collar,
        threshold=0.5,
        num_jobs=num_jobs
    )
    results['collar_based']['error_rate50'] = error_rate['dog']

    # collar based best f1
    best_f1, _, __, best_threshold, ___ = collar_based.best_fscore(
        scores=seds_pred,
        ground_truth=seds_ground_truth,
        onset_collar=collar,
        offset_collar=collar,
        min_recall=best_f1_min_recall,
        num_jobs=num_jobs
    )
    results['collar_based']['best_f1_score'] = best_f1['dog']
    results['collar_based']['best_f1_threshold'] = best_threshold['dog']

    # collar based 0.5 f1
    f1, precision, recall, ___ = collar_based.fscore(
        scores=seds_pred,
        ground_truth=seds_ground_truth,
        threshold=0.5,
        onset_collar=collar,
        offset_collar=collar,
        num_jobs=num_jobs
    )
    results['collar_based']['f1_score50'] = f1['dog']
    results['collar_based']['precision50'] = precision['dog']
    results['collar_based']['recall50'] = recall['dog']

    # collar based best recall
    _, recalls, scores, __ = collar_based.precision_recall_curve(
        scores=seds_pred,
        ground_truth=seds_ground_truth,
        onset_collar=collar,
        offset_collar=collar,
        num_jobs=num_jobs
    )['dog']
    max_recall = max(recalls)
    max_recall_thr = -1.0
    for i in range(len(recalls)):
        if (max_recall - recalls[i]) < 1e-9:
            max_recall_thr = max(max_recall_thr, scores[i])
    results['collar_based']['best_recall'] = max_recall
    results['collar_based']['best_recall_threshold'] = max_recall_thr

    # ---------------------------- collar based end ----------------------------

    # ---------------------------- intersection based start ----------------------------

    # intersection based best error rate
    best_error_rate, _, __, best_threshold, ___ = intersection_based.best_error_rate(
        scores=seds_pred,
        ground_truth=seds_ground_truth,
        dtc_threshold=dtc_threshold,
        gtc_threshold=gtc_threshold,
        num_jobs=num_jobs
    )
    results['intersection_based']['best_error_rate'] = best_error_rate['dog']
    results['intersection_based']['best_er_threshold'] = best_threshold['dog']

    # # intersection based 0.5 error rate
    # error_rate, _, __, ___ = intersection_based.error_rate(
    #     scores=seds_pred,
    #     ground_truth=seds_ground_truth,
    #     dtc_threshold=dtc_threshold,
    #     gtc_threshold=gtc_threshold,
    #     threshold=0.5,
    #     num_jobs=num_jobs
    # )
    # results['intersection_based']['error_rate50'] = error_rate['dog']

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

    # # intersection based 0.5 f1
    # f1, precision, recall, ___ = intersection_based.fscore(
    #     scores=seds_pred,
    #     ground_truth=seds_ground_truth,
    #     threshold=0.5,
    #     dtc_threshold=dtc_threshold,
    #     gtc_threshold=gtc_threshold,
    #     num_jobs=num_jobs
    # )
    # results['intersection_based']['f1_score50'] = f1['dog']
    # results['intersection_based']['precision50'] = precision['dog']
    # results['intersection_based']['recall50'] = recall['dog']

    # intersection based best recall
    _, recalls, scores, __ = intersection_based.precision_recall_curve(
        scores=seds_pred,
        ground_truth=seds_ground_truth,
        dtc_threshold=dtc_threshold,
        gtc_threshold=gtc_threshold,
        num_jobs=num_jobs
    )['dog']
    max_recall = max(recalls)
    max_recall_thr = -1.0
    for i in range(len(recalls)):
        if (max_recall - recalls[i]) < 1e-9:
            max_recall_thr = max(max_recall_thr, scores[i])
    results['intersection_based']['best_recall'] = max_recall
    results['intersection_based']['best_recall_threshold'] = max_recall_thr

    # intersection based best recall
    _, single_class_psds, __, ___ = intersection_based.psds(
        scores=seds_pred,
        ground_truth=seds_ground_truth,
        audio_durations=seds_durations,
        dtc_threshold=dtc_threshold,
        gtc_threshold=gtc_threshold,
        num_jobs=num_jobs
    )
    results['intersection_based']['psds'] = single_class_psds['dog']

    # ---------------------------- intersection based end ----------------------------
    return results


def show_result(root):
    with open(os.path.join(root, 'train_info.txt'), 'r', encoding='utf-8') as file:
        results = [json.loads(line.strip()) for line in file.readlines() if len(line.strip()) > 0]
    train_results = [r for r in results if 'cur_epoch_loss' in r.keys()]
    test_results = [r for r in results if r['mode'] == 'test']

    epochs = [r['epoch'] for r in train_results]
    plt.figure(dpi=200)
    plt.plot(epochs, [r['lr'] for r in train_results], c='red', label='lr', marker='o')
    plt.plot(epochs, [r['cur_epoch_loss'] for r in train_results], c='yellow', label='loss', marker='s')
    plt.title('Train')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()

    steps = [r['step'] for r in test_results]
    sed_results = [r['seds_result'] for r in test_results]
    segment_based_results = [r['segment_based'] for r in sed_results]
    collar_based_results = [r['collar_based'] for r in sed_results]
    intersection_based_results = [r['intersection_based'] for r in sed_results]

    plt.figure(dpi=200)
    plt.plot(steps, [r['best_error_rate'] for r in segment_based_results], c='red', label='ERBest', marker='o')
    plt.plot(steps, [r['error_rate50'] for r in segment_based_results], c='red', label='ER50', marker='s')
    plt.plot(steps, [r['best_f1_score'] for r in segment_based_results], c='yellow', label='F1Best', marker='o')
    plt.plot(steps, [r['f1_score50'] for r in segment_based_results], c='yellow', label='F150', marker='s')
    plt.plot(steps, [r['best_recall'] for r in segment_based_results], c='blue', label='RBest', marker='o')
    plt.plot(steps, [r['recall50'] for r in segment_based_results], c='blue', label='R50', marker='s')
    plt.plot(steps, [r['precision50'] for r in segment_based_results], c='green', label='P50', marker='s')
    plt.title('Segment based')
    plt.xlabel('step')
    plt.legend()
    plt.show()

    plt.figure(dpi=200)
    plt.plot(steps, [r['best_error_rate'] for r in collar_based_results], c='red', label='ERBest', marker='o')
    plt.plot(steps, [r['error_rate50'] for r in collar_based_results], c='red', label='ER50', marker='s')
    plt.plot(steps, [r['best_f1_score'] for r in collar_based_results], c='yellow', label='F1Best', marker='o')
    plt.plot(steps, [r['f1_score50'] for r in collar_based_results], c='yellow', label='F150', marker='s')
    plt.plot(steps, [r['best_recall'] for r in collar_based_results], c='blue', label='RBest', marker='o')
    plt.plot(steps, [r['recall50'] for r in collar_based_results], c='blue', label='R50', marker='s')
    plt.plot(steps, [r['precision50'] for r in collar_based_results], c='green', label='P50', marker='s')
    plt.title('Collar based')
    plt.xlabel('step')
    plt.legend()
    plt.show()

    plt.figure(dpi=200)
    plt.plot(steps, [r['best_error_rate'] for r in intersection_based_results], c='red', label='ERBest', marker='o')
    plt.plot(steps, [r['best_f1_score'] for r in intersection_based_results], c='yellow', label='F1Best', marker='o')
    plt.plot(steps, [r['best_recall'] for r in intersection_based_results], c='blue', label='RBest', marker='o')
    plt.plot(steps, [r['psds'] for r in intersection_based_results], c='green', label='PSDS', marker='s')
    plt.title('Intersection based')
    plt.xlabel('step')
    plt.legend()
    plt.show()






if __name__ == '__main__':
    root = '/home2/xingyuan/animal_language/v2/out/second_cut/panns/alldata_thr0.05_offset1_0-0.01_0.9/[train]_2024-05-28_07-57-22'
    show_result(root)
