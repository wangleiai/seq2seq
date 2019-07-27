# seq2seq
基于pytorch实现seq2seq和attention模型的机器翻译，

## 安装要求
* pytorch 1.1.0
* torchtext 0.4.0
* spacy
* tensorboard
* tensorboardX
* nltk

## 使用说明
    如果使用自己的数据集的时候，需要把param.py中的TEST=False，数据文件的格式:
    data/train.cn data/train.en data/val.cn data/val.en data/test.cn data/test.en 

### train
    修改param.py中的参数,运行train.py，

### evalute
    修改param.py中的参数,device必需设置为cpu,运行evalute.py

### 效果
    src：  我以为那花痴把狗安置好后
    pred：  i thought that was dog

    src：  我觉得，你们跳舞的时候...
    pred：  i think , you dancing ...

    src：  因为那是最重要的事，
    pred：  because that 's the important thing ,

    src：  你认识我- 他妈妈。
    pred：  you know me . - his mother .

    src：  他在牢里被打。
    pred：  he was beaten in prison .

    src：  孟斐斯医院，这是26号车。
    pred：  memphis , this is 26 .

    src：  在瓶子里塞了几块布，一点燃，嘭！
    pred：  put a in bottle , boom !

    src：  是的。我欠你。
    pred：  yes . i owe you .

    src：  在黛娜岸将谜案大起底.
    pred：  to the mystery the .

    src：  据说她逃学了并且离家出走了。
    pred：  said she and away


### 缺点
    1. 没有设置学习率调整策略
    2. teacher_forcing_ratio需要手动调整
    ......
    
## 参考
* https://github.com/bentrevett/pytorch-seq2seq
* https://machinelearningmastery.com/calculate-bleu-score-for-text-python/
* https://github.com/jadore801120/attention-is-all-you-need-pytorch
