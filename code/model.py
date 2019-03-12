import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, num_layers=1):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(self.vocab_size, self.emb_size)
        self.lstm = nn.LSTM(self.emb_size, self.hidden_size, self.num_layers)

    def forward(self, x, hidden=None):
        embedding = self.embedding(x)
        embedding = embedding.permute(1, 0 , 2)
        out, hidden = self.lstm(embedding)

        return out, hidden


class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, output_size,  num_layers=1):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(self.vocab_size, self.emb_size)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(self.emb_size, self.hidden_size, self.num_layers)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x, hidden):
        embedding = self.embedding(x) # (bs, len, seqlen) len =1
        embedding = self.relu(embedding)
        embedding = embedding.permute(1, 0, 2) # (len, bs, seqlen) len =1
        out, hidden = self.lstm(embedding, hidden)
        out = out[-1, :, : ] # 得到最后一个time_step的输出值
        out = out.squeeze(0)  # (bs, seqlen)
        out = self.fc(out)  # (bs, output_size)

        return F.softmax(out), hidden



# en = Encoder(12,2 ,3)
# print(en)
# de = Decoder(12, 2, 3, 4)
# print(de)

