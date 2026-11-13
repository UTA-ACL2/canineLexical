import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy


class FrameBce_masked(nn.Module):
    def __init__(self, hop_length):
        super().__init__()
        self.hop_length = hop_length

    def forward(self, framewise_pred, framewise_target, origin_length):
        """"""
        batch_ori_length = (origin_length // self.hop_length) + 1
        loss_total = 0.
        for i in range(batch_ori_length.shape[0]):
            ori_len = int(batch_ori_length[i])
            _loss = binary_cross_entropy(framewise_pred.clamp(1e-7, 1.)[i, :ori_len], framewise_target[i, :ori_len])
            loss_total += _loss
        return loss_total / batch_ori_length.shape[0]
