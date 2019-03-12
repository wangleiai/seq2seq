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

    def forward(self, x, hidden, context_vextor=None):
        # print('1 ',x.size())
        embedding = self.embedding(x) # (bs, len, emb_size) len =1
        if context_vextor is not None:
            # print('a')
            # print(embedding.size())
            # print(context_vector.size())
            embedding = embedding + context_vextor # (bs, len, emb_size)
        # print(embedding.size())
        embedding = self.relu(embedding)
        embedding = embedding.permute(1, 0, 2) # (len, bs, emb_size) len =1
        out, hidden = self.lstm(embedding, hidden)
        out = out[-1, :, : ] # 得到最后一个time_step的输出值
        out = out.squeeze(0)  # (bs, seqlen)
        out = self.fc(out)  # (bs, output_size)

        return F.softmax(out), hidden


# en = Encoder(12,2 ,3)
# print(en)
# de = Decoder(12, 2, 3, 4)
# print(de)


class Attention(nn.Module):
    def __init__(self, encoder_emb_size, encoder_output_size, decoder_output_size, decoder_emb_size):
        super(Attention, self).__init__()
        self.encoder_emb_size = encoder_emb_size
        self.encoder_output_size = encoder_output_size
        self.decoder_output_size = decoder_output_size
        self.decoder_emb_size = decoder_emb_size

        self.fc = nn.Linear(2*self.encoder_output_size, self.decoder_emb_size)

    def forward(self, decoder_output_hidden, encoder_outputs):
        decoder_output_hidden_mean = torch.mean(decoder_output_hidden, dim=0, keepdim=True).permute(1, 0 ,2) # (bs, 1, hidden_size)
        encoder_outputs = encoder_outputs.permute(1, 2, 0)  # (bs,hiddensize, seqlen)
        score = torch.bmm(decoder_output_hidden_mean, encoder_outputs) # (bs,1, seqlen)
        score = F.softmax(score, 2) # (bs,1, seqlen)
        encoder_outputs = torch.mul(encoder_outputs, score) # (bs,hiddensize, seqlen)
        context_vector = torch.sum(encoder_outputs, 2) # (bs,hiddensize)
        context_vector = context_vector.unsqueeze(1) # (bs,1,hiddensize)
        # print(context_vector.size())
        # print(decoder_output_hidden_mean.size())
        context_vector = torch.cat((decoder_output_hidden_mean, context_vector), 2) # (bs,1,2*hiddensize)
        context_vector = context_vector.squeeze(1) # (bs,2*hiddensize)
        context_vector = self.fc(context_vector).unsqueeze(1) # (bs,1,decoder_emb_size)
        # print(context_vector.size())
        return context_vector


if __name__ == '__main__':

    emb_size = 10
    hidden_size = 6
    decoder_output_size = 100
    vocab = 300

    encoder = Encoder(vocab, emb_size,hidden_size)
    decoder = Decoder(vocab, emb_size,hidden_size,decoder_output_size)
    atten = Attention(emb_size, hidden_size, hidden_size, emb_size)

    # bs = 4
    # en_input = torch.Tensor([[1,2,3],
    #                          [7,8,9]])
    # de_input = torch.Tensor([[1,2,3],
    #                          [7,8,9]])
    # en_input = Variable(en_input).long()
    # de_input = Variable(de_input).long()
    # out, h = encoder(en_input)
    # context_vector = None
    # for i in range(2):
    #     d_out, d_h = decoder(en_input,h, context_vector)
    #     # print(d_h[0].size())
    #     context_vector = atten(d_h[0], out)


    # for i in range(2):
    #


    # # bs = 4
    # emb_size = 5
    # hidden_size = 6
    # output_size = 7
    # vocab = 300
    #
    # en_input = torch.Tensor([[1,2,3],
    #                          [7,8,9]])
    # de_input = torch.Tensor([[1,2,3],
    #                          [7,8,9]])
    # de_tar = torch.Tensor([[1,2,3],
    #                          [7,8,9]])
    #
    # encoder = Encoder(vocab, emb_size, hidden_size, 2)
    # decoder = Decoder(vocab, emb_size, hidden_size, output_size, 2)
    #
    # en_input = Variable(en_input).long()
    # de_input = Variable(de_input).long()
    # de_tar = Variable(de_tar).long()
    #
    # out, h = encoder(en_input)
    # decoder(de_input,h)
    # print(en_input.size())
    # print(out.size())
    # print(h[0].size())
    # print(torch.mean(de_tar.float(), dim=0).size())
    # print(torch.mean(h[0].float(), dim=0, keepdim=True).size())
    #
    # # de_input = torch.Tensor([[1,2,3],
    # #                          [7,8,9]])
    # # de_tar = torch.Tensor([[1,2,3],
    # #                          [7,8,9]])
    # atten = Attention(2,3,4,5)
    # print(atten)