import torch
from torchtext.datasets import  Multi30k
from torchtext.data import Field, BucketIterator

import spacy
import random

import params

# 随机的种子
SEED = 1024
# 每次产生的随机结果一样
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

spacy_de = spacy.load('de')
# spacy_en = spacy.load('en')
spacy_en = spacy.load('en_core_web_sm')
def tokenize_de(text):
    '''
    把德语句子分词，并把句子反转。有论文说把源语言的句子反转会提高准确率。
    :param text:德语句子，例如：Ein Mann mit einem orangefarbenen Hut, der etwas anstarrt.
    :return:['.', 'anstarrt', 'etwas', 'der', ',', 'Hut', 'orangefarbenen', 'einem', 'mit', 'Mann', 'Ein']
    '''
    return [tok.text for tok in spacy_de.tokenizer(text)][::-1]

def tokenize_en(text):
    '''
    把英文句子分词
    :param text:英文句子，例如：A man in an orange hat starring at something.
    :return:['A', 'man', 'in', 'an', 'orange', 'hat', 'starring', 'at', 'something', '.']
    '''
    return [tok.text for tok in spacy_en.tokenizer(text)]

def get_data():
    SRC = Field(tokenize = tokenize_de,init_token = '<sos>',eos_token = '<eos>',pad_token='<pad>',unk_token='<unk>',lower = True)
    TRG = Field(tokenize = tokenize_en,init_token = '<sos>',eos_token = '<eos>',pad_token='<pad>',unk_token='<unk>',lower = True)

    train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'), fields = (SRC, TRG))

    print("train: {}".format(len(train_data.examples)))
    print("valid: {}".format(len(valid_data.examples)))
    print("test: {}".format(len(test_data.examples)))

    SRC.build_vocab(train_data, min_freq = params.MIN_FREQ)
    TRG.build_vocab(train_data, min_freq = params.MIN_FREQ)

    print("源语言词表大小: {}".format(len(SRC.vocab)))
    print("目标语言词表大小: {}".format(len(TRG.vocab)))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=params.BATCH_SIZE,
        device=device)

    return train_iterator, valid_iterator, test_iterator, SRC, TRG

if __name__ == '__main__':
    get_data()