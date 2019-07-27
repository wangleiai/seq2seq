import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from model import base_model, atten_model

from nltk.translate.bleu_score import sentence_bleu
import numpy as np

import params
from translate import translate

def get_bleu(reference, candidate):
    '''
    参考： https://machinelearningmastery.com/calculate-bleu-score-for-text-python/
    计算句子的bleu
    :param reference: 正确的句子 [['this', 'is', 'a', 'test'], ['this', 'is' 'test']]
    :param candidate: 预测的句子 ['this', 'is', 'a', 'test']
    :return:bleu值
    '''
    score = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
    return score

def get_data(path):
    '''
    读取文本
    :param path:文本路径
    :return: list
    '''
    with open(path, mode="r", encoding='utf-8') as f:
        texts = []
        for line in f:
            line = line.strip("\n")
            texts.append(line)
    return texts

def main(test_src_path, test_trg_path):
    checkpoint = torch.load(params.MODEL_PATH)
    SRC = checkpoint["SRC"]
    TRG = checkpoint["TRG"]
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
    elif params.MODEL_TYPE == 2:
        pass
    else:
        print("params.MODEL_TYPE error")
        exit(1)

    # 从文件中读取src和trg
    test_src_data = get_data(test_src_path)
    test_trg_data = get_data(test_trg_path)
    assert len(test_src_data)==len(test_trg_data),"测试src数据和trg数据不匹配"

    trg = torch.from_numpy(np.array([TRG.vocab.stoi['<sos>']])) # trg [1] <sos>是开始标志
    trg = trg.unsqueeze(0) # trg [bs, seq_len]=[1, 1]
    print("trg_size ", trg.size())
    trg_eos_idx = TRG.vocab.stoi['<eos>']

    total_bleu = 0.0
    for idx, src in enumerate(test_src_data):
        src = SRC.preprocess(src)
        # print(src)
        src = SRC.process([src], device=None)  # src [seq_len, 1]
        # print(src.size())
        src = src.permute([1, 0]) # src [1, seq_len]
        # print(src.size())
        outpus = translate(model, src, trg, trg_eos_idx)
        print(test_trg_data[idx])
        # print(TRG.process(test_trg_data[idx]))
        print(TRG.process([TRG.preprocess(test_trg_data[idx])]).permute([1, 0]))
        print(outpus.max(2)[1])

        # print(TRG.vocab.itos[5892])
        exit(0)

if __name__ == '__main__':
    main(params.TEST_SRC_PATH, params.TEST_TRG_PATH)