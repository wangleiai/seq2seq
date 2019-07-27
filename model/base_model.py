import torch
import torch.nn as nn
import torch.nn.functional as F

import random

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers, drop=0.5, bid=False):
        '''
        :param input_dim: 和词标大小一致
        :param emb_dim: 词向量的维度
        :param hidden_dim: 隐藏结点的维度
        :param n_layers: lstm的层数
        :param drop: dropout的比例
        :param bid: 是否双向
        '''
        super().__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.drop = drop
        self.bid = bid

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.dropout = nn.Dropout(drop)
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=hidden_dim, num_layers=n_layers, dropout=drop, bidirectional=bid)

    def forward(self, src):  # src [bs, seq_len]
        embedded = self.embedding(src)  # embedded [bs, seq_len, emb_dim]
        embedded = self.dropout(embedded)  # embedded [bs, seq_len, emb_dim]
        embedded = embedded.permute([1, 0, 2])  # embedded [seq_len, bs, emb_dim]
        outputs, (h_s, c_s) = self.lstm(embedded)
        # output [seq_len, bs, hidden_dim * n_directions]
        # hidden  [n_layers * n directions, bs, hidden_dim]
        # cell  [n_layers * n directions, bs, hidden_dim]
        return h_s, c_s

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers, drop=0.5, bid=False):
        super().__init__()
        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.drop = drop
        self.bid = bid

        self.embdding = nn.Embedding(output_dim, emb_dim)
        self.dropout = nn.Dropout(drop)
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=hidden_dim, num_layers=n_layers, dropout=drop, bidirectional=bid)
        if bid:
            self.linear = nn.Linear(in_features=hidden_dim*2, out_features=output_dim)
        else:
            self.linear = nn.Linear(in_features=hidden_dim, out_features=output_dim)
    def forward(self, inp, h_s, c_s):  # inp [bs]
        inp = inp.unsqueeze(1)  # trg [bs, 1]
        embeded = self.embdding(inp)  # embeded [bs, 1, emb_dim]
        embeded = self.dropout(embeded)  # embeded [bs, 1, emb_dim]
        embeded = embeded.permute([1, 0, 2])  # embeded [1, bs, emb_dim]
        out, (h_s, c_s) = self.lstm(embeded, (h_s, c_s))
        # output [1, bs, hidden_dim * n_directions]
        # hidden  [n_layers * n directions, bs, hidden_dim]
        # cell  [n_layers * n directions, bs, hidden_dim]
        out = out.squeeze(0)  #  out [bs, hidden_dim * n_directions]
        # print("out: ", out.size())
        out = self.linear(out) #  out [bs, output_dim]
        return out, h_s, c_s

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0, inference=False, trg_eos_idx=None):
        '''
        :param src: [bs, seq_len]
        :param trg: 在训练时 trg:[bs, seq_len], 在测试时 bs:[bs, 1]
        :param teacher_forcing_ratio: 使用teach force的比例
        :return: 预测值 [bs, seq_len]
        '''
        batch_size = trg.shape[0]
        if not inference:
            max_len = trg.shape[1]
        else:
            max_len = 30
        trg_vocab_size = self.decoder.output_dim
        # 用来存储预测时值
        outputs = torch.zeros(batch_size, max_len, trg_vocab_size).to(self.device)

        h_s, c_s = self.encoder(src)
        inp = trg[:, 0] # trg [bs]
        # print("inp")
        for t_s in range(1, max_len):
            out, h_s, c_s = self.decoder(inp, h_s, c_s)
            # 把预测结果保存到outputs
            outputs[:,t_s,:] = out
            if random.random() < teacher_forcing_ratio:
                inp = trg[:, t_s]
            else:
                inp = out.max(1)[1]  # [bs]
                if trg_eos_idx is not None:
                    if inp.item() == trg_eos_idx:
                        break
        return outputs


if __name__ == '__main__':
    enc = Encoder(10, 20, 10, 2, 0.5)
    dec = Decoder(10, 20, 10, 2, 0.5)

    model = Seq2Seq(enc, dec, "cuda:0").to("cuda:0")
