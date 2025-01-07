import numpy as np
import pandas as pd
from collections import Counter
import jieba

def cut_sentence(s):
    """定义分词函数"""
    return list(jieba.cut(s))

def data_process(filepath):
    content = pd.read_csv(filepath, sep='\t', header=None)
    content.columns = ['sentence', 'label']
    content['words'] = content['sentence'].apply(lambda s: ' '.join(cut_sentence(s)))
    content['words'] = content['words'].apply(lambda s: ' '.join(s.split())[:30])
    return content

if __name__ == '__main__':
    train_filepath = '../data/train.txt'
    dev_filepath = '../data/dev.txt'
    test_filepath = '../data/test.txt'

    train_data = data_process(train_filepath)
    dev_data = data_process(dev_filepath)
    test_data = data_process(test_filepath)
    
    train_data.to_csv('../data/train_new.csv')
    test_data.to_csv('../data/test_new.csv')
    dev_data.to_csv('../data/dev_new.csv')    