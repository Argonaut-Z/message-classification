import torch
import os
from transformers import RobertaConfig, BertTokenizer, RobertaForSequenceClassification
from datetime import datetime

class Config(object):
    def __init__(self, dataset):      
        self.model_name = 'roberta'
        self.data_path = '/mnt/workspace/message_classification/BERT/data/'
        self.train_path = self.data_path + "train.txt"  # 训练集
        self.dev_path = self.data_path + "dev.txt"  # 验证集
        self.test_path = self.data_path + "test.txt"    # 测试集
        self.class_list = [
            x.strip() for x in open(self.data_path + "class.txt").readlines()
        ]   # 类别名单
        self.save_path = '/mnt/workspace/message_classification/BERT/cache'
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        
        # 获取当前日期和时间，格式为 YYYY-MM-DD_HH-MM
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
        self.save_path += "/" + self.model_name + "-" + current_time + ".pt" # 模型训练结果
        self.save_path2 = './cache/' + self.model_name + '_quantize' + ".pt"   # 剪枝后的模型
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.require_improvement = 1000 # 若超过1000batch效果还没有提升，则提前结束训练
        self.num_classes = len(self.class_list) # 类别数
        self.num_epochs = 5 # epoch数
        self.batch_size = 128   # mini-batch大小
        self.pad_size = 32  # 每句话处理成的长度（短补长截）
        self.learning_rate = 2e-5   # 学习率
        self.hidden_size = 768
        self.roberta_path = '/mnt/workspace/message_classification/BERT/pretrain_model/chinese-roberta-wwm-ext'
        self.roberta_config = RobertaConfig.from_pretrained(self.roberta_path + '/config.json')
        self.tokenizer = BertTokenizer.from_pretrained(self.roberta_path)


if __name__ == '__main__':
    dataset = "toutiao"
    config = Config(dataset)
    print("config.data_path:", config.data_path)
    print("config.save_path:", config.save_path)
    print("config.device:", config.device)
    print("Tokenizer test:", config.tokenizer("今天的天气真好"))

