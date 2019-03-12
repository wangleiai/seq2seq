import nltk
import pkuseg
import langconv
from collections import Counter
import numpy as np
import json

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

def write_to_file(en_dict=None, cn_dict=None):
    en_json = json.dumps(en_dict)
    cn_json = json.dumps(cn_dict)

    with open('../data/en_vocab.json', mode='a', encoding='utf-8') as f:
        f.write(en_json)
        f.close()
    with open('../data/cn_vocab.json', mode='a', encoding='utf-8') as f:
        f.write(cn_json)
        f.close()

if __name__ == '__main__':
    en_vocab_size = 4000
    cn_vocab_size = 5000

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

    # 把字典存到文件里面
    write_to_file(en_dict, cn_dict)
    print('写入文件')