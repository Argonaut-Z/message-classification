import torch
import time
from datetime import timedelta
import os
import pickle as pkl
from transformers import BertTokenizer
from tqdm import tqdm


UNK, PAD, CLS = "[UNK]", "[PAD]", "[CLS]"  # 特殊符号
MAX_VOCAB_SIZE = 10000  # 词表长度限制

# 定义构建词表的函数
def build_vocab(file_path, tokenizer, max_size, min_freq):
    vocab_dic = {}
    with open(file_path, "r", encoding='utf-8') as f:
        for line in tqdm(f):
            line = line.strip()
            if not line:
                continue
            content = line.split("\t")[0]   # 获取文本内容
            for word in tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        vocab_list = sorted(
            [_ for _ in vocab_dic.items() if _[1]>=min_freq], key=lambda x: x[1], reverse=True
        )[:max_size]
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}   # word_to_idx 字典
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic


def build_dataset(config):
    """
    根据提供的配置文件加载训练集、验证集和测试集，并对数据进行预处理。
    Args:
        config: 包含配置信息的对象，包含路径、分词器、pad_size 等。
    Returns:
        train, dev, test: 预处理后的训练集、验证集和测试集。
    """
    def load_dataset(path, pad_size=32):
        """
        加载并预处理单个数据集。
        Args:
            path: 数据文件路径。
            pad_size: 序列的最大长度。如果小于 pad_size，则进行填充；如果大于，则截断。
        Returns:
            contents: 包含 (token_ids, label, seq_len, mask) 的数据列表。
        """
        contents = []
        with open(path, "r", encoding='utf-8') as f:
            for line in tqdm(f):  # tqdm 用于显示进度条
                line = line.strip()
                if not line:
                    continue  # 跳过空行
                # 数据格式假定为 '文本\t标签'
                content, label = line.split('\t')
                token = config.tokenizer.tokenize(content)  # 分词操作
                token = [CLS] + token  # 在序列开头添加特殊标记 [CLS]
                seq_len = len(token)   # 序列长度
                token_ids = config.tokenizer.convert_tokens_to_ids(token)   # 转换为 ID
                mask = []   # 构建 mask 和 padding
                if pad_size:
                    if len(token) < pad_size:
                        # 如果序列长度不足 pad_size，填充 0
                        mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                        token_ids += [0] * (pad_size - len(token))
                    else:
                        # 如果序列长度超过 pad_size，进行截断
                        mask = [1] * pad_size
                        token_ids = token_ids[:pad_size]
                        seq_len = pad_size

                # 将处理后的数据添加到结果列表
                contents.append((token_ids, int(label), seq_len, mask))

        return contents

    # 加载训练集、验证集和测试集
    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return train, dev, test


def build_dataset_textCNN(config):
    tokenizer = lambda x: [y for y in x]    # char-level
    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, "rb"))
    else:
        vocab = build_vocab(config.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(vocab, open(config.vocab_path, "wb"))
    
    def load_dataset(path, pad_size=32):
        contents = []
        with open(path, "r", encoding="utf-8") as f:
            for line in tqdm(f):
                line = line.strip()
                if not line:
                    continue
                content, label = line.split("\t")
                words_line = []
                token = tokenizer(content)
                seq_len = len(token)
                if pad_size:
                    if seq_len < pad_size:
                        token.extend([PAD] * (pad_size - seq_len))
                    else:
                        token = token[:pad_size]
                        seq_len = pad_size
                # word_to_idx
                for word in token:
                    words_line.append(vocab.get(word, vocab.get(UNK)))
                contents.append((words_line, int(label), seq_len))
        return contents
    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return vocab, train, dev, test

class DatasetIterater(object):
    def __init__(self, batches, batch_size, device, model_name):
        self.batch_size = batch_size
        self.batches = batches
        self.model_name = model_name
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.batch_size != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        if self.model_name == "bert" or self.model_name == "multi_task_bert":
            mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
            return (x, seq_len, mask), y
        if self.model_name == "textCNN":
            return (x, seq_len), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size : len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size : (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device, config.model_name)
    return iter

def get_time_dif(start_time):
    # 获取已使用时间
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('../bert_pretrain')

    # 构建词表
    vocab_dic = build_vocab('../data/train.txt', tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
    print("生成的词表大小:", len(vocab_dic))