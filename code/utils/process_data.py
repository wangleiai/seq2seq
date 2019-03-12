import os
import nltk
import pkuseg
from random import shuffle
# import langconv
from . import langconv
from collections import Counter
# from keras.utils import to_categorical
import numpy as np

def get_lang(data_path):
    '''
    提取中英文句子，数据每一行都按照tab键分割,并返回数据
    :param data_path:  数据存放路径
    :return: 分割出来的数据en, cn
    '''
    en = []
    cn = []
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            e, c = line.strip().split('\t')
            c = langconv.Converter('zh-hans').convert(c)
            en.append(e.lower())
            cn.append(c)
        f.close()
    return en, cn

def en_tokenize(seqtences):
    '''
    把英文句子变为一个一个的词，
    :param seqtences:
    :return:
    '''
    token = []
    for inx, seq in enumerate(seqtences):
        token.append(nltk.word_tokenize(seq))
    return token

def cn_tokenize(seqtences):
    '''
    把中文句子变为一个一个的词,分词
    :param seqtences:
    :return:
    '''
    seg = pkuseg.pkuseg()

    token = []
    for idx, seq in enumerate(seqtences):
        token.append(seg.cut(seq))
    return token

def split_data(en, cn, train_per=0.8, valid_per=0.2, is_sort=True):
    '''
    把数据集按比例分割，并存到路径下面
    :param en: 英文句子数据列表 []
    :param cn: 中文句子数据列表 ['']
    :param train_per:训练数据所占比例
    :param valid_per:验证集所占比例
    :return:
    '''
    # 打乱数据集
    ec = list(zip(en, cn))
    shuffle(ec)
    en[:], cn[:] = zip(*ec)

    # 用下标切分数据集
    t_idx = int(train_per*len(en))
    v_idx = int((train_per+valid_per)*len(en))
    # print(t_idx)
    # print(v_idx)
    en_train_data = en[:t_idx]
    cn_train_data = cn[:t_idx]
    en_valid_data = en[t_idx : v_idx]
    cn_valid_data = cn[t_idx : v_idx]
    # code.interact()

    return en_train_data, cn_train_data, en_valid_data, cn_valid_data

def add_eos_sos(seqs):
    '''
    [['hop', 'in', '.']]->[['sos', 'hop', 'in', '.', 'eos']]
    :param seqs:
    :return:
    '''
    new_seq = []
    for i in range(len(seqs)):
        seq = seqs[i]
        seq.append('eos') # 开始标记
        seq.insert(0, 'sos') # 结束标记
        new_seq.append(seq)
    return new_seq

def word_to_idx(seqs, dt):
    '''
    [['sos', 'hop', 'in', '.', 'eos']]->[[2, 18, 19, 5, 3]]
    :param seqs:
    :param dt:
    :return:
    '''
    idx_seq = []
    for idx, seq in enumerate(seqs):
        idxs = []
        for i, s in enumerate(seq):
            if s not in dt.keys():
                idxs.append(dt['unk'])
            else :
                idxs.append(dt[s])
        # idxs = [dt[s] for i, s in enumerate(seq)]
        idx_seq.append(idxs)
    return idx_seq

def idx_to_word(seqs, rv_dt):
    '''
    [[2, 18, 19, 5, 3]] -> [['sos', 'hop', 'in', '.', 'eos']]
    :param seqs:
    :param rv_dt: 字典
    :return:
    '''
    word_seq = []
    for idx, numbers in enumerate(seqs):
        seq = []
        for num in numbers:
            if num==0: # 0代表unk， 不翻译
                seq.append('unk')
                continue
            seq.append(rv_dt[num])
        word_seq.append(seq)
    return word_seq

def build_dict(sentences, max_size=3000):
    '''
    为句子建立词典
    :param sentences: 一个列表[['hop', 'in', '.'], ['i', 'lost', '.'], ['i', 'quit', '.']]
    :max_size: 表示字典大小
    :return:
    '''
    dt = {'unk':0, 'pad':1 ,'sos':2, 'eos':3}
    size = 4
    word_count = Counter()
    for sentence in sentences:
        for s in sentence:
            word_count[s] += 1
    ls = word_count.most_common(max_size-size)
    print(ls)
    for idx, word in enumerate(ls):
        dt[word[0]] = size
        size += 1

    # for idx, seq in enumerate(sentences):
    #     for word in seq:
    #         if word not in dt.keys():
    #             dt[word] = size
    #             size += 1
    return dt

def pad_unk(seqs):
    max_len = 0
    # print('seqs: ', seqs)
    for idx, seq in enumerate(seqs):
        if max_len<len(seq):
            max_len = len(seq)

    seqs_pad = []
    # 添加pad
    for idx, seq in enumerate(seqs):
        for add in range(max_len-len(seq)):
            seq.append(1) # 1在字典中是pad
        seqs_pad.append(seq)
    return seqs_pad

# def idx_to_onehot(seqs, cates):
#     onehot_seq = []
#     for words in seqs:
#         w = []
#         for word in words:
#             onthot = to_categorical(word, cates)
#             w.append(onthot)
#         onehot_seq.append(w)
#     return onehot_seq

def sort_data(data1, data2):
    # 对数据排序， 遍历一遍数据，拿到每一行的长度，然后对长度进行排序,

    data_len = []
    for idx in range(len(data1)):
           data_len.append(len(data1[idx]))

    data = [ (le, d1, d2) for le, d1, d2 in zip(data_len,data1, data2)]
    data.sort()

    data1 = [d1 for le, d1, d2 in data]
    data2 = [d2 for le, d1, d2 in data]
    return data1, data2

def get_batchdata(src_data,src_vocab_size,tar_data, tar_vocab_size,batch_size=4):
    src_data, tar_data = sort_data(src_data, tar_data)

    # print(src_data[0:5])
    # print(tar_data[0:5])
    #
    # exit()

    encoder_inputs = []
    decoder_inputs = []
    outputs = []
    count = 0
    while True:
        for idx, s in enumerate(src_data):

            encoder_inputs.append(s)
            decoder_inputs.append(tar_data[idx])
            # print(tar_data[idx])
            tar = []
            for idx, data in enumerate(tar_data[idx]):
                if idx!=0:
                    tar.append(data)
            tar.append(1)
            # tar = tar_data[idx][1:]
            # tar = tar.append(1)
            # print("tar: ", tar)
            outputs.append(tar)
            count += 1
            if count==batch_size:
                # 拓展到相同长度
                encoder_inputs = pad_unk(encoder_inputs)
                decoder_inputs = pad_unk(decoder_inputs)
                outputs = pad_unk(outputs)
                while True:
                    yield np.array(encoder_inputs), np.array(decoder_inputs), np.array(outputs)
                encoder_inputs = []
                decoder_inputs = []
                outputs = []
                count = 0

def get_dict(filename, max_size=3000):
    en, cn = get_lang("../cmn.txt")
    # print(en[10:13])
    # 分词
    print("tokenizer")
    en_seq = en_tokenize(en)
    print(en_seq[10:13])
    cn_seq = cn_tokenize(cn)
    print(cn_seq[10:13])

    # 建立词典
    en_dict = build_dict(en_seq, max_size)
    cn_dict = build_dict(cn_seq, max_size)
    print("build dict")
    print("英文字典长度: ", len(en_dict.keys()))
    print('中文字典长度: ', len(cn_dict.keys()))
    return en_dict, cn_dict

if __name__ == '__main__':
    en_vocab_size = 3000
    cn_vocab_size = 4000

    # 加载数据集
    print('load dataset')
    en, cn = get_lang("../data/cmn.txt")
    print(en[10:13])

    # 分词
    print("tokenizer")
    en_seq = en_tokenize(en)
    # print(en_seq[10:13])
    cn_seq = cn_tokenize(cn)
    # print(cn_seq[10:13])

    # 建立词典
    en_dict = build_dict(en_seq, max_size=en_vocab_size)
    cn_dict = build_dict(cn_seq, max_size=cn_vocab_size)
    print("build dict")
    print("英文字典长度: ", len(en_dict.keys()))
    print('中文字典长度: ', len(cn_dict.keys()))
    # print(en_dict)
    # print(cn_dict)

    # 给句子加上 sos 和eos
    en_seq = add_eos_sos(en_seq)
    cn_seq = add_eos_sos(cn_seq)
    print("add sos and eos")
    print(en_seq[10:13])
    print(cn_seq[10:13])

    # word -> idx
    en_idx = word_to_idx(en_seq, en_dict)
    cn_idx = word_to_idx(cn_seq, cn_dict)
    print("word -> idx")
    print(en_idx[10:13])
    print(cn_idx[10:13])

    # idx -> onehot
    # !!!!!!!!!!!!!!!!!! 这里要记得改en_idx[10:13] ->en_idx 在这之前还要padding
    # en_onehot = idx_to_onehot(en_idx[10:13], len(en_dict.keys()))
    # cn_onehot = idx_to_onehot(cn_idx[10:13], len(cn_dict.keys()))
    # print('transform to onehot')
    # print(np.array(en_onehot).shape)
    # print(np.array(cn_onehot).shape, cn_onehot)

    # idx -> word
    # 将之前id->word字典反转
    en_rv_dt = dict(zip(en_dict.values(), en_dict.keys()))
    cn_rv_dt = dict(zip(cn_dict.values(), cn_dict.keys()))
    en_word = word_to_idx(en_idx[10:13], en_rv_dt)
    cn_word = word_to_idx(cn_idx[10:13], cn_rv_dt)
    print(en_word)
    print(cn_word)

    # 切分数据集
    en_train_data, cn_train_data, en_valid_data, cn_valid_data = split_data(en_idx, cn_idx)
    print("split dataset ")
    print('训练集数量: ',len(en_train_data))
    print('验证集数量: ',len(en_valid_data))
    print(en_train_data[10:13])
    print(cn_train_data[10:13])
    print(en_valid_data[0:3])
    print(cn_valid_data[0:3])

    print('get batch data')
    for d1, d2,_ in get_batchdata(en_valid_data, 3000, cn_valid_data, 4000):
        print(d1)
        print(idx_to_word(list(d1), en_rv_dt))
        print(d2)
        print(idx_to_word(list(d2), cn_rv_dt))
        break