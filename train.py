import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.optim import lr_scheduler

import time
import math
from tensorboardX import SummaryWriter

from model import base_model, atten_model
import dataSet
import dataSetN
import params
from utils import labelSmooth

def cal_loss(pred, gold, smoothing, pad_index):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(pad_index)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later

        non_pad_mask = gold.ne(pad_index)
        n_word = non_pad_mask.sum().item()
        loss = loss/n_word
    else:

        loss = F.cross_entropy(pred, gold, ignore_index=pad_index)
        loss = loss/n_word
    return loss

s_writer = SummaryWriter()
global train_steps
train_steps = 0


def train(model, iterator, is_smoothing, optimizer, pad_idx, clip=1):
    model.train()
    total_loss = 0.0
    # tem_trg = None #  后面写入graph的时候使用
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg
        src = src.permute([1, 0])
        trg = trg.permute([1, 0])
        # tem_trg = trg
        optimizer.zero_grad()
        outputs = model(src, trg, 0.5)
        # trg  [bs, seq_len]
        # output [bs, seq_len, output_dim]
        # 去掉trg中<sos>和去掉outputs中第一行无效数据
        outputs = outputs[:, 1:, :].contiguous().view(-1, outputs.size(-1))
        trg = trg[:, 1:].contiguous().view(-1)
        # outputs = outputs[:,1:].view(-1, outputs.shape[-1])
        # trg = trg[1:].view(-1)

        loss = cal_loss(outputs, trg, is_smoothing, pad_idx)
        # print(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item()
        global train_steps
        train_steps += 1
        s_writer.add_scalar("train/step_loss", loss.item(), train_steps)
        # break
    #  写graph到tensorborad中
    # s_writer.add_graph(model, (src, tem_trg))
    return total_loss / len(iterator)


def valid(model, iterator, is_smoothing, pad_idx):
    model.eval()
    total_loss = 0.0
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg
        src = src.permute([1, 0])
        trg = trg.permute([1, 0])

        outputs = model(src, trg, 0.5)
        # trg  [bs, seq_len]
        # output [bs, seq_len, output_dim]
        # 去掉trg中<sos>和去掉outputs中第一行无效数据
        outputs = outputs[:, 1:, :].contiguous().view(-1, outputs.size(-1))
        trg = trg[:, 1:].contiguous().view(-1)

        loss = cal_loss(outputs, trg, is_smoothing, pad_idx)
        total_loss += loss.item()
    return total_loss / len(iterator)


def main(epoches=10):
    if params.TEST:
        train_iterator, valid_iterator, test_iterator, SRC, TRG = dataSet.get_data()
    else:
        train_iterator, valid_iterator, test_iterator, SRC, TRG = dataSetN.get_data()
    if params.MODEL_TYPE == 0:
        enc = base_model.Encoder(len(SRC.vocab), params.ENC_EMB_DIM, params.HID_DIM, params.N_LAYERS, params.ENC_DROPOUT,
                                 params.BID)
        dec = base_model.Decoder(len(TRG.vocab), params.ENC_EMB_DIM, params.HID_DIM, params.N_LAYERS, params.DEC_DROPOUT,
                                 params.BID)

        model = base_model.Seq2Seq(enc, dec, params.DEVICE).to(params.DEVICE)
    elif params.MODEL_TYPE == 1:
        attn = atten_model.Atten(params.HID_DIM, params.HID_DIM, params.N_LAYERS, bid=params.BID)
        enc = atten_model.Encoder(len(SRC.vocab), params.ENC_EMB_DIM, params.HID_DIM, params.N_LAYERS, params.ENC_DROPOUT, bid=params.BID)
        dec = atten_model.Decoder(len(TRG.vocab), params.DEC_EMB_DIM, params.HID_DIM, params.HID_DIM, params.N_LAYERS, attn, bid=params.BID)
        model = atten_model.Seq2Seq(enc, dec, params.DEVICE).to(params.DEVICE)

    else:
        print("params.MODEL_TYPE error")
        exit(1)

    # 自定义初始化权重
    def init_weights(m):
        for name, param in m.named_parameters():
            # print(name)
            nn.init.uniform_(param.data, -0.08, 0.08)

    model.apply(init_weights)

    # 计算参数 torch.numel() 返回一个tensor变量内所有元素个数
    def count_train_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def count_total_parameters(model):
        return sum(p.numel() for p in model.parameters())

    print("训练参数数量：{}".format(count_train_parameters(model)))
    print("总参数数量：{}".format(count_total_parameters(model)))

    PAD_IDX = SRC.vocab.stoi['<pad>']
    print("pad token：{}".format(PAD_IDX))

    optimizer = optim.Adam(model.parameters(), lr=params.LR)
    
    best_model = None
    for epoch in range(1, epoches + 1):
        train_loss = train(model, train_iterator, params.LABEL_SMOOTH, optimizer, PAD_IDX, clip=1)
        print("epoch:{} train loss : {}".format(epoch, train_loss))
        valid_loss = valid(model, valid_iterator, params.LABEL_SMOOTH, PAD_IDX)
        print("epoch:{} valid loss : {}".format(epoch, valid_loss))

        s_writer.add_scalar('train/epoch_loss', train_loss, epoch)
        s_writer.add_scalar('valid/epoch_loss', valid_loss, epoch)

        if best_model is None or best_model > valid_loss:
            best_model = valid_loss
            # 保存模型的同时，保存下SRC和TRG，等到翻译时不必重新生成
            torch.save({'model': model.state_dict(),
                        'SRC': SRC,
                        'TRG': TRG}, params.MODEL_PATH)
        torch.save({'model': model.state_dict(),
                    'SRC': SRC,
                    'TRG': TRG}, "models/1.pt")

if __name__ == '__main__':
    main(19)

