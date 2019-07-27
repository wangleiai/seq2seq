import torch
from torchtext.data import Field, BucketIterator, TabularDataset
from torchtext.datasets import TranslationDataset
import spacy
import random
import jieba

import params

# 随机的种子
SEED = 1024
# 每次产生的随机结果一样
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# spacy_en = spacy.load('en')
spacy_en = spacy.load('en_core_web_sm')

def tokenize_en(text):
    '''
    把英文句子分词
    :param text:英文句子，例如：A man in an orange hat starring at something.
    :return:['A', 'man', 'in', 'an', 'orange', 'hat', 'starring', 'at', 'something', '.']
    '''
    return [tok.text for tok in spacy_en.tokenizer(text)]

def tokenize_cn(text):
    return [i for i in jieba.cut(text)][::-1]

def get_data(path='data/'):
    SRC = Field(tokenize = tokenize_cn,init_token = '<sos>',eos_token = '<eos>',pad_token='<pad>',unk_token='<unk>',lower = True)
    TRG = Field(tokenize = tokenize_en,init_token = '<sos>',eos_token = '<eos>',pad_token='<pad>',unk_token='<unk>',lower = True)

    train_data, valid_data, test_data = TranslationDataset.splits(path=path, train= 'train',validation='val', test='test',
                                                                  exts=('.cn', '.en'), fields=(SRC, TRG))

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