import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
import json


from utils import process_data
from attenModel import Encoder, Decoder,Attention


# 载入词典
with open('./data/cn_vocab.json', mode='r', encoding='utf-8') as f:
    cn_dict = dict(json.load(f))
with open('./data/en_vocab.json', mode='r', encoding='utf-8') as f:
    en_dict = dict(json.load(f))
# print(type(en_dict))
en, cn = process_data.get_lang("./data/cmn.txt")

# 分词
en_seq = process_data.en_tokenize(en)
cn_seq = process_data.cn_tokenize(cn)
# 给句子加上 sos 和eos
en_seq = process_data.add_eos_sos(en_seq)
cn_seq = process_data.add_eos_sos(cn_seq)
# word -> idx
en_idx = process_data.word_to_idx(en_seq, en_dict)
cn_idx = process_data.word_to_idx(cn_seq, cn_dict)
print('process data finished')


# 做测试，没有分出来验证集
en_train_data, cn_train_data, en_valid_data, cn_valid_data = process_data.split_data(en_idx, cn_idx, train_per=0.9, valid_per=0.1)
print('split data finished')
# print(len(en_train_data))

batch_size = 128
IS_CUDA = True
emb_size = 200
hidden_size = 256
num_layers = 2

def train(epoches=1):
    encoder = Encoder(4000, emb_size, hidden_size, num_layers)
    decoder = Decoder(5000, emb_size, hidden_size, 4000, num_layers)
    atten = Attention(emb_size, hidden_size, hidden_size, emb_size)
    if IS_CUDA:
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        atten = atten.cuda()

    # criterion = nn.CrossEntropyLoss()
    encoder_optim = optim.Adam(encoder.parameters(), 0.0001)
    decoder_optim = optim.Adam(decoder.parameters(), 0.0001)
    atten_optim = optim.Adam(atten.parameters(), 0.0001)
# 这里记得改 en_valid_data -》 en_train_data
    for epoch in range(epoches):
        step = 0
        total_loss = 0
        for encoder_inputs, decoder_inputs, decoder_tars in process_data.get_batchdata(en_train_data, 3000, cn_train_data, 4000, batch_size=batch_size):
            if step>=len(en_train_data)//batch_size:
                break
            step += 1
            loss = 0
            # embedding 需要long类型
            encoder_inputs = Variable(torch.from_numpy(encoder_inputs).long())
            decoder_inputs = Variable(torch.from_numpy(decoder_inputs).long())
            decoder_tars = Variable(torch.from_numpy(decoder_tars).long())

            if IS_CUDA:
                encoder_inputs = encoder_inputs.cuda()
                decoder_inputs = decoder_inputs.cuda()
                decoder_tars = decoder_tars.cuda()

            encoder_outputs, hidden = encoder(encoder_inputs)
            # print('size: ', decoder_tars.size(1))
            s = ''
            context_vector = None
            for i in range(decoder_tars.size(1)):
                # print(decoder_inputs[:, i].unsqueeze(1).size())
                # print(decoder_inputs[:, i].unsqueeze(1))  # torch.Size([4])
                # if context_vector is not None:

                decoder_outputs, hidden = decoder(decoder_inputs[:, i].unsqueeze(1), hidden, context_vector)
                context_vector = atten(hidden[0], encoder_outputs)
                decoder_tar = decoder_tars[:, i]
                # print(decoder_outputs.size())
                s += str(decoder_outputs[0, :].data.topk(1)[1].item())+' '
                loss += F.cross_entropy(decoder_outputs, decoder_tar.long())
            print(s)
            print(decoder_tars[0, :].cpu().numpy())
            encoder_optim.zero_grad()
            decoder_optim.zero_grad()
            atten_optim.zero_grad()
            loss.backward()
            encoder_optim.step()
            decoder_optim.step()
            atten_optim.step()
            total_loss += loss.item() / decoder_tars.size(1)
            print('epoch:',epoch,' step:',step,  ' train_loss:', loss.item()/decoder_tars.size(1))
        total_loss /=(len(en_train_data) // batch_size)
        if epoch %100==0:
            encoder_model_path = './attenModel/'+'encoder'+ str(epoch)+'_loss_'+ str(total_loss) + '.pkl'
            torch.save(encoder, encoder_model_path)
            decoder_model_path = './attenModel/'+'decoder' + str(epoch) + '_loss_' + str(total_loss) + '.pkl'
            torch.save(decoder, decoder_model_path)
            atten_model_path = './attenModel/' + 'atten' + str(epoch) + '_loss_' + str(total_loss) + '.pkl'
            torch.save(decoder, atten_model_path)
train(10000)