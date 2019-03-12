import json
import os

with open('../data/cn_vocab.json', mode='r', encoding='utf-8') as f:
    print(json.load(f))