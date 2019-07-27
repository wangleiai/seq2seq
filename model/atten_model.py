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
        # print(src.is_cuda)
        embedded = self.embedding(src)  # embedded [bs, seq_len, emb_dim]
        embedded = self.dropout(embedded)  # embedded [bs, seq_len, emb_dim]
        embedded = embedded.permute([1, 0, 2])  # embedded [seq_len, bs, emb_dim]
        outputs, (h_s, c_s) = self.lstm(embedded)
        # output [seq_len, bs, hidden_dim * n_directions]
        # hidden  [n_layers * n directions, bs, hidden_dim]
        # cell  [n_layers * n directions, bs, hidden_dim]
        return outputs, h_s, c_s


class Atten(nn.Module):
    def __init__(self, enc_hidden_dim,  dec_hidden_dim, n_layers, bid=False):
        super().__init__()
        self.enc_hidden_dim = enc_hidden_dim
        self.dce_hidden_dim = dec_hidden_dim

        self.v = nn.Parameter(torch.rand(dec_hidden_dim))
        if bid:
            self.linear = nn.Linear(dec_hidden_dim*2*n_layers, enc_hidden_dim*2)
            self.attn = nn.Linear(enc_hidden_dim * 2 + dec_hidden_dim*2, dec_hidden_dim)
        else:
            self.linear = nn.Linear(dec_hidden_dim*1*n_layers, enc_hidden_dim*1)
            self.attn = nn.Linear(enc_hidden_dim  + dec_hidden_dim, dec_hidden_dim)

    def forward(self, hidden, enc_outputs):
        # hidden  [n_layers * n directions, bs, hidden_dim]
        # encoder_outputs  [seq_len, bs, hidden_dim * n_directions]
        bs = hidden.size(1)
        src_seq_len = enc_outputs.size(0)

        enc_outputs = enc_outputs.permute([1, 0, 2])
        hidden = hidden.permute([1, 0, 2])  # [bs, n_layers*n_directions, hidden_dim]
        hidden = hidden.contiguous().view(bs, -1)  # [bs, n_layers*n_directions*hidden_dim]
        hidden = self.linear(hidden) # [bs, hidden_dim*n_directions]

        # .repeat(size) 是复制几倍的意思
        hidden = hidden.unsqueeze(1).repeat(1, src_seq_len, 1)  # [bs, seq_len, hidden_dim*n_directions]

        #hidden = [bs, src seq len, hidden_dim*n_directions]
        #encoder_outputs = [bs, src seq len, enc hidden dim * n_directions]
        #  hidden_dim == enc hid dim

        energy = torch.tanh(self.attn(torch.cat((hidden, enc_outputs), dim=2))) # cat: [bs, seq_len, 2*hidden_dim*n_directions]
        # energy [bs, src_seq_len, hidden_dim]
        energy = energy.permute(0, 2, 1) # energy = [bs, hidden_dim, src_seq_len]

        # v = [hidden_dim]
        v = self.v.repeat(bs, 1).unsqueeze(1) # v = [bs, 1, hidden_dim]
        attention = torch.bmm(v, energy).squeeze(1) # attention= [batch size, src len]

        return F.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hidden_dim, dec_hidden_dim, n_layers, atten, drop=0.5, bid=False):
        super().__init__()
        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.en_hidden_dim = enc_hidden_dim
        self.de_hidden_dim = dec_hidden_dim
        self.n_layers = n_layers
        self.atten = atten
        self.drop = drop
        self.bid = bid

        self.embedding = nn.Embedding(output_dim, emb_dim)
        if bid:
            self.lstm = nn.LSTM(input_size=emb_dim+dec_hidden_dim*2, hidden_size=dec_hidden_dim, num_layers=n_layers, dropout=drop, bidirectional=bid)
            self.out = nn.Linear(emb_dim+dec_hidden_dim*2+enc_hidden_dim*2, output_dim)
        else:
            self.lstm = nn.LSTM(input_size=emb_dim+dec_hidden_dim, hidden_size=dec_hidden_dim, num_layers=n_layers, dropout=drop, bidirectional=bid)
            self.out = nn.Linear(emb_dim+dec_hidden_dim+enc_hidden_dim, output_dim)

        self.dropout = nn.Dropout(drop)

    def forward(self,  inp, h_s, c_s, encoder_outputs):
        # inp [bs]
        # encoder_outputs [seq_len, bs, hidden_dim * n_directions]
        # h_s  [n_layers * n directions, bs, hidden_dim]
        # c_s  [n_layers * n directions, bs, hidden_dim]
        inp = inp.unsqueeze(1)  # trg [bs, 1]
        embeded = self.embedding(inp)  #  embeded [bs, 1, emb_dim]
        embeded = self.dropout(embeded)  #  embeded [bs, 1, emb_dim]
        embeded = embeded.permute([1, 0, 2])  #  embeded [1, bs, emb_dim]

        v = self.atten(h_s, encoder_outputs) # v [bs, seq len]
        v = v.unsqueeze(1) # v [bs, 1, seq len]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # encoder_outputs = [bs, seq_len, hidden_dim * n_directions]
        weighted = torch.bmm(v, encoder_outputs)
        # weighted = [bs, 1, hidden_dim * n_directions]
        weighted = weighted.permute(1, 0, 2)
        # weighted = [1, bs, hidden_dim * n_directions]

        lstm_inp = torch.cat((weighted, embeded), dim=2)
        # lstm_inp [1, bs, hidden_dim * n_directions+emb_dim]
        output, (h_s, c_s) = self.lstm(lstm_inp, (h_s, c_s))
        # output [seq_len, bs, hidden_dim * n_directions]
        # hidden  [n_layers * n directions, bs, hidden_dim]
        # cell  [n_layers * n directions, bs, hidden_dim]

        output = output.squeeze(0) # output [bs, hidden_dim * n_directions]
        weighted = weighted.squeeze(0) # weighted [bs, hidden_dim * n_directions]
        embeded = embeded.squeeze(0) # embeded [bs, emb_dim]

        out_inp = torch.cat((output, weighted, embeded), dim=1)
        output = self.out(out_inp)
        return output, h_s, c_s

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.0, inference=False, trg_eos_idx=None):
        #src = [bs, src_seq_len]
        #trg = [bs, trg_seq_len]
        #teacher_forcing_ratio is probability to use teacher forcing

        bs = trg.shape[0]
        if not inference:
            max_len = trg.shape[1]
        else:
            max_len = 30
        trg_vocab_size = self.decoder.output_dim
        # print("trg_vocab_size ", trg_vocab_size)
        # print("bs ", bs)
        # print("max_len ", max_len)
        # 用来存储预测时值
        outputs = torch.zeros(bs, max_len, trg_vocab_size).to(self.device)

        encoder_output, h_s, c_s = self.encoder(src)
        # encoder_output [seq_len, bs, hidden_dim * n_directions]
        # hidden  [n_layers * n directions, bs, hidden_dim]
        # cell  [n_layers * n directions, bs, hidden_dim]

        inp = trg[:, 0] # trg [bs]
        for t_s in range(1, max_len):
            out, h_s, c_s = self.decoder(inp, h_s, c_s, encoder_output)
            # 把预测结果保存到outputs
            # print(out.size())
            # print(outputs.size())
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
    INPUT_DIM = 500
    OUTPUT_DIM = 100
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    ENC_HID_DIM = 512
    DEC_HID_DIM = 512
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5
    N_LAYERS = 2
    BID = True
    attn = Atten(ENC_HID_DIM, DEC_HID_DIM, N_LAYERS, bid=BID)
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, N_LAYERS, ENC_DROPOUT, bid=BID)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM,ENC_HID_DIM, DEC_HID_DIM,N_LAYERS, attn, bid=BID)

    model = Seq2Seq(enc, dec, "cpu:0")
    print(model)