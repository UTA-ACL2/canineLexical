import os
import sys

import librosa
import numpy as np
import pandas as pd
from torch import nn
from torch.nn.functional import pad
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

sys.path.append('../')
sys.path.append('../../')
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import argparse
import bisect
import json
import logging
from sed_scores_eval import *
import os
from datetime import datetime

import torch
from pytorch.models import *
from utils import config
from tqdm import tqdm

from util import set_seed, log_init
from dataset import *
from loss import *
from evaluate import *

"""
nohup python train.py \
      --device 1 \
      --train_positive_labels dog,dog_dog,dog_noise,pant \
      --test_positive_labels dog,dog_dog,dog_noise,pant \
      --train_filter_flag > /bigdrive/xingyuan/v2/out/second_cut/panns/alldata_thr0.05_offset1_0-0.01_0.9/train_filter_all-label.out 2>&1 &

nohup python train.py \
      --device 1 \
      --train_positive_labels dog,dog_dog,dog_noise \
      --test_positive_labels dog,dog_dog,dog_noise \
      --train_filter_flag > /bigdrive/xingyuan/v2/out/second_cut/panns/alldata_thr0.05_offset1_0-0.01_0.9/train_filter_dog-label.out 2>&1 &

nohup python train.py \
      --device 1 \
      --train_positive_labels dog,dog_dog,dog_noise,pant \
      --test_positive_labels dog,dog_dog,dog_noise,pant > /bigdrive/xingyuan/v2/out/second_cut/panns/alldata_thr0.05_offset1_0-0.01_0.9/train_all-label.out 2>&1 &

nohup python train.py \
      --device 1 \
      --train_positive_labels dog,dog_dog,dog_noise \
      --test_positive_labels dog,dog_dog,dog_noise > /bigdrive/xingyuan/v2/out/second_cut/panns/alldata_thr0.05_offset1_0-0.01_0.9/train_dog-label.out 2>&1 &

nohup python train.py \
      --device 1 \
      --train_positive_labels dog,dog_dog,dog_noise,pant \
      --test_positive_labels dog,dog_dog,dog_noise \
      --train_filter_flag > /bigdrive/xingyuan/v2/out/second_cut/panns/alldata_thr0.05_offset1_0-0.01_0.9/train_filter_alld-label.out 2>&1 &

nohup python train.py \
      --device 1 \
      --train_positive_labels dog,dog_dog,dog_noise,pant \
      --test_positive_labels dog,dog_dog,dog_noise > /bigdrive/xingyuan/v2/out/second_cut/panns/alldata_thr0.05_offset1_0-0.01_0.9/train_alld-label.out 2>&1 &

nohup python train.py \
      --device 1 \
      --train_positive_labels dog,dog_dog,dog_noise \
      --data_window_size 6.0 \
      --data_stride 3.0 \
      --train_batch_size 64 \
      --test_by_step 10 \
      --test_positive_labels dog,dog_dog,dog_noise > /bigdrive/xingyuan/v2/out/second_cut/panns/alldata_thr0.05_offset1_0-0.01_0.9/train_dog-label6.out 2>&1 &

nohup python train.py \
      --device 0 \
      --train_positive_labels dog,dog_dog,dog_noise \
      --data_window_size 3.0 \
      --data_stride 1.5 \
      --train_batch_size 128 \
      --test_by_step 10 \
      --test_positive_labels dog,dog_dog,dog_noise > /bigdrive/xingyuan/v2/out/second_cut/panns/alldata_thr0.05_offset1_0-0.01_0.9/train_dog-label3.out 2>&1 &

nohup python train.py \
      --device 0 \
      --train_positive_labels dog,dog_dog,dog_noise \
      --data_window_size 15.0 \
      --data_stride 7.5 \
      --train_batch_size 32 \
      --test_by_step 10 \
      --test_positive_labels dog,dog_dog,dog_noise > /bigdrive/xingyuan/v2/out/second_cut/panns/alldata_thr0.05_offset1_0-0.01_0.9/train_dog-label15.out 2>&1 &

nohup python train.py \
      --device 1 \
      --train_positive_labels dog,dog_dog,dog_noise \
      --data_window_size 10.0 \
      --data_stride 5.0 \
      --train_batch_size 32 \
      --test_by_step 10 \
      --test_positive_labels dog,dog_dog,dog_noise > /bigdrive/xingyuan/v2/out/second_cut/panns/alldata_thr0.05_offset1_0-0.01_0.9/train_dog-label10.out 2>&1 &

----------------------------------------------------------
nohup python train.py \
      --device 1 \
      --train_positive_labels dog,dog_dog,dog_noise \
      --data_window_size 10.0 \
      --data_stride 5.0 \
      --train_batch_size 128 \
      --test_by_step 3 \
      --test_positive_labels dog,dog_dog,dog_noise > /bigdrive/xingyuan/v2/out/second_cut/panns/alldata_thr0.05_offset1_0-0.01_0.9/train_dog-label10-128.out 2>&1 &


"""


cur_root = os.path.dirname(os.path.abspath(__file__))
checkpoint_path = os.path.join(os.path.dirname(cur_root), 'checkpoints/Cnn14_DecisionLevelMax_mAP=0.385.pth')

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
                        default='dog,dog_dog,dog_noise,pant')
    parser.add_argument('--test_positive_labels', type=str, required=False,
                        default='dog,dog_dog,dog_noise,pant')

    # dataset
    parser.add_argument('--sr', type=int, required=False,
                        default=32000)
    parser.add_argument('--data_window_size', type=float, required=False,
                        default=20.0)
    parser.add_argument('--data_stride', type=float, required=False,
                        default=10.0)
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




def get_model(device, require_grad_layers):
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


    # 微调模型，更换最后一层全连接层
    model.fc_audioset = nn.Linear(in_features=2048, out_features=1, bias=True)
    init_layer(model.fc_audioset)

    # 冻结不需要训练的层
    for name, child in model.named_children():
        if name in require_grad_layers:
            for param in child.parameters():
                param.requires_grad = True
        else:
            for param in child.parameters():
                param.requires_grad = False

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


def train(*,
          args, model, device, hop_size,
          train_dataloader,
          test_dataloader,
          info_save_path,
          frames_per_second,
          checkpoints_save_root):
    loss_fn = FrameBce_masked(hop_length=hop_size)
    optimizer = Adam(model.parameters(), lr=args.learning_rate,
                     betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1, min_lr=1e-7)

    step = 0
    epoch = 0
    while (step < args.steps) and (epoch < args.epochs):
        epoch_loss = []
        for train_batch_dict in train_dataloader:
            # torch.cuda.empty_cache()
            train_waveform = train_batch_dict['waveform'].to(torch.float32)
            train_target = train_batch_dict['target'].to(torch.float32)
            train_origin_length = train_batch_dict['origin_length']

            optimizer.zero_grad()
            model.train()
            loss_fn.train()
            train_batch_output = model(train_waveform.to(device), None)
            train_framewise_pred = train_batch_output['framewise_output'][:, :, 0]
            train_framewise_target = train_target[:, ::hop_size].to(device)
            train_pad_torch = train_framewise_target[:, -1:].repeat(1, train_framewise_pred.shape[1] - train_framewise_target.shape[1])
            train_framewise_target = torch.cat((train_framewise_target, train_pad_torch), dim=1)
            loss = loss_fn(framewise_pred=train_framewise_pred, framewise_target=train_framewise_target, origin_length=train_origin_length.cpu().detach().numpy())
            epoch_loss.append(loss.cpu().detach().item())

            loss.backward()
            optimizer.step()

            train_info = {
                'mode': 'train',
                'epoch': epoch,
                'step': step,
                'lr': optimizer.param_groups[0]['lr'],
                'train_batch_size': len(train_origin_length),
                'train_loss': loss.cpu().detach().item()
            }
            logging.info(train_info)
            with open(info_save_path, 'a+', encoding='utf-8') as file:
                file.write(json.dumps(train_info, ensure_ascii=False))
                file.write('\n')

            if (step % args.test_by_step) == 0 and (step != 0):
                checkpoints_path = os.path.join(checkpoints_save_root, f'checkpoint_{epoch}_{step}.pth')
                torch.save(model.state_dict(), checkpoints_path)
                test_info = {key: value for key, value in train_info.items()}
                test_info['mode'] = 'test'

                test_pred = {}
                for test_batch_dict in test_dataloader:
                    test_waveform = test_batch_dict['waveform'].to(torch.float32)
                    with torch.no_grad():
                        model.eval()
                        loss_fn.eval()
                        test_batch_output = model(test_waveform.to(device), None)
                    test_framewise_pred = test_batch_output['framewise_output'][0, :, 0]
                    test_id = test_batch_dict['name'][0]
                    test_pred[test_id] = test_framewise_pred.cpu().detach().numpy()
                seds_result = get_seds_pred(test_results=test_pred,
                                            label_root=args.test_labels_root,
                                            frames_per_second=frames_per_second,
                                            audios_root=args.audios_root,
                                            sr=args.sr,
                                            test_positive_labels=args.test_positive_labels)
                test_info['seds_result'] = seds_result

                logging.info(test_info)
                with open(info_save_path, 'a+', encoding='utf-8') as file:
                    file.write(json.dumps(test_info, ensure_ascii=False))
                    file.write('\n')

            step += 1
        cur_epoch_loss = np.mean(epoch_loss)
        cur_info = {
            'mode': 'train',
            'epoch': epoch,
            'step': step,
            'lr': optimizer.param_groups[0]['lr'],
            'cur_epoch_loss': cur_epoch_loss,
        }
        logging.info(cur_info)
        with open(info_save_path, 'a+', encoding='utf-8') as file:
            file.write(json.dumps(cur_info, ensure_ascii=False))
            file.write('\n')
        lr_scheduler.step(cur_epoch_loss)
        epoch += 1


if __name__ == '__main__':
    args = get_args()
    args.train_positive_labels = args.train_positive_labels.split(',')
    args.test_positive_labels = args.test_positive_labels.split(',')
    set_seed(seed=args.seed, speed_up=False)
    name = f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    save_root = os.path.join(args.save_root, f'[train]_{name}')

    checkpoints_save_root = os.path.join(save_root, 'checkpoints')
    os.makedirs(checkpoints_save_root, exist_ok=True)

    info_save_path = os.path.join(save_root, 'train_info.txt')
    log_save_path = os.path.join(save_root, f'train.log')
    log_init(log_save_path)
    logging.info(args)

    device = torch.device(f'cuda:{args.device}') if torch.cuda.is_available() and (args.device > -1) else \
             torch.device('cpu')

    require_grad_layers = [
        'spectrogram_extractor',
        'logmel_extractor',
        'spec_augmenter',
        'bn0',
        'conv_block1',
        'conv_block2',
        'conv_block3',
        'conv_block4',
        'conv_block5',
        'conv_block6',
        'fc1',
        'fc_audioset',
    ]
    model, frames_per_second, hop_size = get_model(device=device, require_grad_layers=require_grad_layers)
    logging.info(f'Model loaded(GPU={args.device}).')

    train_dataset = DogDataset(args=args, mode='train')
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.train_batch_size,
                                  shuffle=True, num_workers=args.train_num_workers, pin_memory=False)

    test_dataset = DogDataset(args=args, mode='test')
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.test_batch_size,
                                 shuffle=False, num_workers=args.test_num_workers, pin_memory=False)
    logging.info('Dataset loaded.')

    train(args=args,
          model=model,
          device=device,
          hop_size=hop_size,
          train_dataloader=train_dataloader,
          test_dataloader=test_dataloader,
          info_save_path=info_save_path,
          frames_per_second=frames_per_second,
          checkpoints_save_root=checkpoints_save_root)
