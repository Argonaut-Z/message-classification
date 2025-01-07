# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from datetime import datetime


class TextCNNConfig(object):
    def __init__(self, dataset):
        self.model_name = "textCNN"
        self.data_path = "/mnt/workspace/message_classification/BERT/data/"
        self.train_path = self.data_path + "train.txt"  # 训练集
        self.dev_path = self.data_path + "dev.txt"  # 验证集
        self.test_path = self.data_path + "test.txt"  # 测试集
        self.class_list = [x.strip() for x in open(self.data_path+"class.txt", encoding="utf-8").readlines()]
        self.vocab_path = self.data_path + "vocab.pkl"  # 词表
        self.save_path = '/mnt/workspace/message_classification/BERT/cache'
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        
        # 获取当前日期和时间，格式为 YYYY-MM-DD_HH-MM
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
        self.save_path += "/" + self.model_name + ".pt" # 模型训练结果
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设备

        self.dropout = 0.5  # 随机失活
        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)  # 类别数
        self.n_vocab = 100  # 词表大小，在运行时赋值
        self.num_epochs = 50  # epoch数
        self.batch_size = 128  # mini-batch大小
        self.pad_size = 32  # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3  # 学习率
        self.embed = 300  # 字向量维度
        self.filter_sizes = (2, 3, 4)  # 卷积核尺寸
        self.num_filters = 512  # 卷积核数量(channels数)
  

if __name__ == '__main__':
    dataset = "toutiao"
    config = Config(dataset)
    print("config.data_path:", config.data_path)
    print("config.vocab_path:", config.vocab_path)
    print("config.filter_sizes:", config.filter_sizes)
    print("config.device", config.device)
    