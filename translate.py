import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn


def translate(model, src, trg, trg_eos_idx):
    '''
    :param model:
    :param src:[bs, seq_len] bs=1
    :param trg:[bs, 1]
    :param trg_pad_idx:
    :return: outputs
    '''
    model.eval()
    trg = trg.long()
    outputs = model(src, trg, inference=True, trg_eos_idx=trg_eos_idx)

    return outputs