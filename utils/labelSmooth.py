from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

import params

# 参考 pytorch-opennmt 中onmt\utils\loss.py
class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value, device=params.DEVICE)
        one_hot[self.ignore_index] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """

        model_prob = self.one_hot.repeat(target.size(0), 1)

        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

        return F.kl_div(output, model_prob, reduction='sum')


class LabelSmoothing(nn.Module):

    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):

        super(LabelSmoothing, self).__init__()

        self.criterion = nn.KLDivLoss(size_average=False)

        self.padding_idx = padding_idx

        self.confidence = 1.0 - smoothing

        self.smoothing = smoothing

        self.size = size

        self.true_dist = None



    def forward(self, x, target):

        assert x.size(1) == self.size

        true_dist = x.data.clone()

        true_dist.fill_(self.smoothing / (self.size - 2))

        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        true_dist[:, self.padding_idx] = 0

        mask = torch.nonzero(target.data == self.padding_idx)

        if mask.dim() > 0:

            true_dist.index_fill_(0, mask.squeeze(), 0.0)

        self.true_dist = true_dist

        return self.criterion(x, Variable(true_dist, requires_grad=False))