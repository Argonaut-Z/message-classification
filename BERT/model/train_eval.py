import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils.data_helpers import get_time_dif
from torch.optim import AdamW
from tqdm import tqdm
import math
import logging

def train(config, model, train_iter, dev_iter):
    start_time = time.time()
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01
        },
        {
            "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
    loss_fn = nn.CrossEntropyLoss()  # 损失函数定义一次

    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升，用于判断是否早停

    model.train()
    for epoch in range(config.num_epochs):
        print(f"Epoch [{epoch + 1}/{config.num_epochs}]")
        for i, (trains, labels) in enumerate(tqdm(train_iter)):
            # 解构输入，确保符合 forward() 方法
            context, seq_len, mask = trains
            outputs = model((context, seq_len, mask))

            model.zero_grad()  # 梯度清零
            loss = loss_fn(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            if total_batch % 200 == 0 and total_batch != 0:
                # 每 200 轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predict = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predict)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    torch.save(model.state_dict(), config.save_path)  # 保存模型
                    dev_best_loss = dev_loss  # 更新最佳验证集loss
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ""
                time_dif = get_time_dif(start_time)
                msg = f"Iter: {total_batch}, Train Loss: {loss.item():.2f}, Train Acc: {train_acc:.2%}, Val Loss: {dev_loss:.2f}, Val Acc: {dev_acc:.2%}, Time: {time_dif} {improve}"
                print(msg)
                model.train()

            total_batch += 1

            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过指定batch没有下降，提前结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break


def test(config, model, test_iter):
    # model.load_state_dict(torch.load(config.save_path))
    # 采用量化模型进行推理时需要关闭
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    
    msg = "Test Loss: {0:>5.2}, Test Acc: {1:>6.2%}"
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    # 采用量化模型进行推理时需要关闭
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predict = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predict)
    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)