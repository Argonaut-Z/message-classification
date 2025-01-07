# coding: UTF-8
import numpy as np
import torch
import time
from config.config_bert import Config
from config.config_textCNN import TextCNNConfig
from model.train_eval import train, test
from model.train_eval_textCNN import train_kd
from importlib import import_module
import argparse
from utils.data_helpers import build_dataset
from utils.data_helpers_textCNN import build_dataset_textCNN, build_iterator, get_time_dif

parser = argparse.ArgumentParser(description="Chinese Text Classification")
parser.add_argument("--task", type=str, required=True, help="choose a task: trainbert, or train_kd")
args = parser.parse_args()


if __name__ == "__main__":
    dataset = "toutiao"

    if args.task == "train_kd":
        model_name = "bert_classification"
        bert_module = import_module("model." + model_name)
        bert_config = Config(dataset)
        
        model_name = "textCNN_classification"
        cnn_module = import_module("model." + model_name)
        cnn_config = TextCNNConfig(dataset)

        np.random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        torch.backends.cudnn.deterministic = True  # 保证每次结果一样

        # 构建bert数据集，因为只需要训练结果作为软目标，这里不需要dev_iter和test_iter
        bert_train_data, _, _ = build_dataset(bert_config)
        bert_train_iter = build_iterator(bert_train_data, bert_config)

        # 构建cnn数据集
        vocab, cnn_train_data, cnn_dev_data, cnn_test_data = build_dataset_textCNN(cnn_config)
        cnn_train_iter = build_iterator(cnn_train_data, cnn_config)
        cnn_dev_iter = build_iterator(cnn_dev_data, cnn_config)
        cnn_test_iter = build_iterator(cnn_test_data, cnn_config)
        cnn_config.n_vocab = len(vocab)

        print("Data loaded, now load teacher model")
        # 加载训练好的teacher模型
        bert_model = bert_module.Model(bert_config).to(bert_config.device)
        bert_model.load_state_dict(torch.load("/mnt/workspace/message_classification/BERT/cache/bert.pt"))

        
        # 加载student模型
        cnn_model = cnn_module.Model(cnn_config).to(cnn_config.device)

        print("Teacher and student models loaded, start training")
        train_kd(bert_config, cnn_config, bert_model, cnn_model,
                 bert_train_iter, cnn_train_iter, cnn_dev_iter, cnn_test_iter)
