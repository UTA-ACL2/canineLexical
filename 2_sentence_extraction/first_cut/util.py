import json
import logging
import os
import random

import numpy as np
import torch


def set_seed(seed, speed_up):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现。
    if not speed_up:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False # 禁用基准测试，可能会降低性能


def log_init(filename: str = None, console_level=logging.INFO, file_level=logging.INFO):
    logger = logging.getLogger()
    formatter = logging.Formatter(
        fmt='%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s: %(message)s',
        datefmt='[%Y-%m-%d %H:%M:%S]'
    )
    logger.setLevel(logging.DEBUG)  # level one

    # output to file
    if filename is not None and filename.endswith('.log'):
        filename = filename.replace('\\', '/')
        file_handler = logging.FileHandler(filename=filename, mode='w+', encoding='utf-8')
        file_handler.setLevel(level=file_level)  # level two
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # output to console
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level=console_level)  # level two
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


def show(path):
    with open(path, 'r', encoding='utf-8') as file:
        data = [json.loads(line.strip()) for line in file.readlines() if len(line.strip()) > 0]
    audio_count = len(data)
    clip_times = []
    for a in data:
        for c in a['clips']:
            clip_times.append(c['end'] - c['start'])
    print(f'clip count: {len(clip_times)}')
    print(f'clip all time: {sum(clip_times)}')
    print(f'clip avg time: {sum(clip_times) / len(clip_times)}')
    print(f'clip min time: {min(clip_times)}')
    print(f'clip max time: {max(clip_times)}')
    print(f'audio count: {audio_count}')
    print(f'audio avg count: {len(clip_times) / audio_count}')
    print(f'audio avg time: {sum(clip_times) / audio_count}')

