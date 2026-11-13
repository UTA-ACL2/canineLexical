import json
import os

from matplotlib import pyplot as plt
from tqdm import tqdm
import librosa


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
    # root = '/home2/xingyuan/animal_language/v2/out/second_cut/panns/alldata_thr0.05_offset1_0-0.01_0.9/[train]_2024-05-28_07-57-29'
    # show_result(root)

    # root = '/home/xingyuan/animal_language2/data/train_second_cut/alldata_thr0.05_offset1_0-0.01_0.9/audios'
    # total = 0.
    # for file in tqdm(os.listdir(root)):
    #     wav, sr = librosa.load(os.path.join(root, file), mono=True)
    #     total += wav.shape[0] / sr
    # print(total)

    root = '/home/xingyuan/animal_language2/data/train_second_cut/alldata_thr0.05_offset1_0-0.01_0.9/train_labels'
    root2 = ''
