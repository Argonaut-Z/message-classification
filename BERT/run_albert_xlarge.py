import time
import torch
import numpy as np
from model.train_eval import train, test
from config.config_albert_xlarge import Config
from importlib import import_module
import argparse
from utils.data_helpers import build_dataset, build_iterator, get_time_dif


parser = argparse.ArgumentParser(description="Chinese Text Classification")
parser.add_argument("--model", type=str, required=True, help="choose a model: albert")
args = parser.parse_args()


if __name__ == "__main__":
    dataset = "toutiao"  # 数据集
    if args.model == "albert":
        model_name = "albert_classification"
        x = import_module("model." + model_name)
        config = Config(dataset)
        np.random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        torch.backends.cudnn.deterministic = True  # 保证每次结果一样

        print("Loading data for Bert Model...")
        train_data, dev_data, test_data = build_dataset(config)
        train_iter = build_iterator(train_data, config)
        dev_iter = build_iterator(dev_data, config)
        test_iter = build_iterator(test_data, config)

        model = x.Model(config).to(config.device)
        train(config, model, train_iter, dev_iter)
        test(config,model, test_iter)