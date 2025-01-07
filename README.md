# 项目说明

## 信息自动投递项目背景介绍

今日头条是字节跳动旗下的核心产品之一，其推荐系统依赖于强大的自然语言处理（NLP）技术。短文本分类是该推荐系统的重要组成部分，用于自动将新闻内容按类别分类，为推荐引擎提供支持。

**目标**：
将新闻标题划分为固定的 10 个类别（如财经、娱乐、体育等），根据分类结果推荐更符合用户兴趣的内容，提升用户体验和平台收益。

该项目旨在构建一个基于多种模型的文本分类系统，用于实现信息的精准自动投递。系统采用多种机器学习和深度学习方法，包括 `Random Forest` 、`FastText` 和 `BERT`，以满足不同场景下的分类需求。

**意义**：

1. **提高推荐精准度**：分类结果用于匹配用户兴趣，提升点击率和停留时长。
2. **增强数据管理效率**：自动分发新闻至对应频道，优化内容组织。
3. **创造商业价值**：增加订阅量、广告收益和用户粘性。

该项目是 NLP 技术在推荐系统中的典型应用，是字节跳动生态的重要技术支撑。

- 基于`Randomforest`的基线模型
- 基于`Fasttext`的基线模型
- 基于`BERT`的文本分类模型

## 1. 环境准备

### 1.1 配置环境

在`conda`环境下创建虚拟环境并安装如下必要库：

```shell
conda create -n bert_classification python=3.10.14 -y
conda activate bert_classification
pip install ipykernel==6.29.5

pip install torch==2.3.1 -i https://pypi.tuna.tsinghua.edu.cn/simple/ 
pip install transformers==4.47.0 -i https://pypi.tuna.tsinghua.edu.cn/simple/ 
pip install numpy==1.26.4 -i https://pypi.tuna.tsinghua.edu.cn/simple/ 
pip install pandas==2.2.3 -i https://pypi.tuna.tsinghua.edu.cn/simple/ 
pip install scikit-learn==1.5.2 -i https://pypi.tuna.tsinghua.edu.cn/simple/ 
pip install jieba==0.42.1 -i https://pypi.tuna.tsinghua.edu.cn/simple/  
pip install icecream==2.1.3 -i https://pypi.tuna.tsinghua.edu.cn/simple/ 
pip install tqdm==4.67.1 -i https://pypi.tuna.tsinghua.edu.cn/simple/ 
pip install Flask==3.1.0 -i https://pypi.tuna.tsinghua.edu.cn/simple/ 
pip install waitress==3.0.2 -i https://pypi.tuna.tsinghua.edu.cn/simple/ (windows)
pip install gunicorn==23.0.0 -i https://pypi.tuna.tsinghua.edu.cn/simple/ (linux)
```

或者根据`requirements.txt`安装必要库：

```shell
pip install torch==2.3.1 transformers==4.47.0 numpy==1.26.4 pandas==2.2.3 scikit-learn==1.5.2 jieba==0.42.1 fasttext==0.9.2 icecream==2.1.3 tqdm==4.67.1 Flask==3.1.0 waitress==3.0.2 gunicorn==23.0.0 
pip install -r requirements.txt
```

在`ModuleScope`云平台下自带上述库，无需再重复安装。

windows系统下安装`fasttext`包，在本地已经下载好对应文件，执行如下命令：

```shell
pip install fasttext_wheel-0.9.2-cp310-cp310-win_amd64.whl
```

### 1.2 命令行添加系统路径

```shell
export PYTHONPATH=/mnt/workspace/message_classification/BERT:$PYTHONPATH
```

## 2. 数据

### 2.1 数据

- 数据集由 **今日头条** 新闻数据构成，包含短文本内容及对应的分类标签。
- 数据分为 **训练集（train.txt）**、**验证集（dev.txt）** 和 **测试集（test.txt）**，以及 **分类标签文件（class.txt）**。

### 2.2 数据格式

- 每行表示一条短文本数据，包含新闻标题及其对应的分类标签，格式如下：

  ```css
  标题 \t 标签
  ```

  - **标题**：新闻标题，为短文本（如 "中华女子学院：本科层次仅1专业招男生"）。
  - **标签**：整数类型，表示对应的新闻类别（如 `3` 表示该新闻属于某一类别）。

```css
中华女子学院：本科层次仅1专业招男生     3
两天价网站背后重重迷雾：做个网站究竟要多少钱    4
东5环海棠公社230-290平2居准现房98折优惠 1
卡佩罗：告诉你德国脚生猛的原因 不希望英德战踢点球       7
```

### 2.3 数据文件内容

1. `class.txt`：包含 10 个分类标签及其描述，每行表示一个类别名称：

   ```css
   finance        # 财经
   realty         # 房地产
   stocks         # 股票
   education      # 教育
   science        # 科学
   society        # 社会
   politics       # 政治
   sports         # 体育
   game           # 游戏
   entertainment  # 娱乐
   ```

2.`train.txt`

- **训练集数据**，包含 **180,000 条样本**。

- 数据格式为：

  ```css
  新闻标题 \t 类别标签
  ```

- 用途：

  - 用于模型的训练，帮助模型学习新闻标题与分类之间的映射关系。

3. `dev.txt`

- **验证集数据**，包含 **10,000 条样本**。

- 数据格式与 `train.txt` 相同：

  ```css
  新闻标题 \t 类别标签
  ```

- 用途：

  - 用于训练过程中评估模型的性能，调整超参数。
  - 通过验证集上的准确率或损失，判断模型是否过拟合或欠拟合。

4. `test.txt`

- **测试集数据**，包含 **10,000 条样本**。

- 数据格式与 `train.txt` 相同：

  ```css
  新闻标题 \t 类别标签
  ```

- 用途：用于评估模型的最终性能。

### 2.4 数据特点

1. **短文本特征**：
   - 数据集中每条样本为短文本，标题长度较短，通常不足一行。
   - 文本内容包含丰富的信息，既有关键字，又有复杂的语义关联。
2. **多分类任务**：
   - 数据分类标签为整数类型，共有 10 个类别。
   - 类别之间互斥，每条数据只能属于一个类别。
3. **类别分布**：
   - 需通过统计进一步确认类别分布是否均衡。
4. **语言特征**：
   - 数据为中文，需采用 NLP 技术进行分词和特征提取。

### 2.5 总结

这份数据为多分类任务提供了一个高质量的数据集，涵盖 **中文短文本、多类别标签** 的特点，是典型的 NLP 应用场景。结合项目背景，分类结果不仅可以用于用户兴趣匹配，还能帮助优化推荐系统逻辑，为字节跳动等相关企业带来商业价值。

## 3. 项目目录结构

以下是`message_classification`文件夹下的目录结构和内容介绍：

```css
message_classification/
├── baseline/                       # 基线模型目录
│   ├── data/                       # 数据文件目录
│   │   ├── class.txt               # 类别定义文件
│   │   ├── dev_fast.txt            # 验证集（快速加载格式）
│   │   ├── dev_fast1.txt           # 另一个验证集文件
│   │   ├── dev.txt                 # 验证集（原始格式）
│   │   ├── dev_new.csv             # 验证集（CSV格式）
│   │   ├── stopwords.txt           # 停用词表
│   │   ├── test_fast.txt           # 测试集（快速加载格式）
│   │   ├── test_fast1.txt          # 另一个测试集文件
│   │   ├── test.txt                # 测试集（原始格式）
│   │   ├── test_new.csv            # 测试集（CSV格式）
│   │   ├── train_fast.txt          # 训练集（快速加载格式）
│   │   ├── train_fast1.txt         # 另一个训练集文件
│   │   ├── train.txt               # 训练集（原始格式）
│   │   ├── train_new.csv           # 训练集（CSV格式）
│   ├── fasttext/                   # FastText相关代码和模型
│   │   ├── app.py                  # FastText服务端代码
│   │   ├── fast_text_baseline1.ipynb  # FastText基线模型版本1
│   │   ├── fast_text_baseline2.ipynb  # FastText基线模型版本2
│   │   ├── fast_text_baseline3.ipynb  # FastText基线模型版本3
│   │   ├── preprocess.ipynb        # 数据预处理代码（分字级别）
│   │   ├── preprocess1.ipynb       # 数据预处理代码（分词级别）
│   │   ├── test.py                 # FastText模型测试代码
│   │   ├── toutiao_fasttext_20241227_110922.bin  # 训练好的FastText模型
│   │   ├── toutiao_fasttext_20241227_110737.bin  # 另一个训练好的FastText模型
│   ├── random_forest/              # 随机森林相关代码和分析
│   │   ├── analysis.ipynb          # 随机森林模型分析代码
│   │   ├── analysis.py             # 随机森林模型分析脚本
│   │   ├── random_forest_baseline1.ipynb  # 随机森林基线模型代码
├── BERT/                           # BERT模型相关代码和数据
│   ├── bert_pretrain/              # 预训练BERT模型文件
│   │   ├── bert_config.json        # BERT配置文件
│   │   ├── pytorch_model.bin       # BERT模型二进制文件
│   │   ├── vocab.txt               # BERT词汇表
│   ├── cache/                      # 缓存目录
│   ├── config/                     # 配置文件目录
│   │   ├── __init__.py             # 配置模块初始化文件
│   │   ├── config.py               # 配置参数定义文件
│   ├── data/                       # 数据文件目录（与baseline的data类似）
│   │   ├── class.txt               # 类别定义文件
│   │   ├── dev.txt                 # 验证集
│   │   ├── stopwords.txt           # 停用词表
│   │   ├── test.txt                # 测试集
│   │   ├── train.txt               # 训练集
│   ├── model/                      # 模型代码目录
│   │   ├── __init__.py             # 模型模块初始化文件
│   │   ├── bert_classification.py  # BERT分类模型定义
│   │   ├── train_eval.py           # 训练与评估函数
│   ├── utils/                      # 工具代码目录
│   │   ├── __init__.py             # 工具模块初始化文件
│   │   ├── data_helpers.py         # 数据处理函数
│   ├── bert_classification.ipynb   # BERT分类模型实验记录
│   ├── run.py                      # 主运行脚本
├── README.md                       # 项目说明文档
├── requirements.txt                # 项目依赖包列表

```

## 4. 详细内容介绍

以下是对项目目录结构的详细说明以及内容的功能和用途。

------

### 4.1 顶层目录

- **`baseline/`**
  - 存放基线模型（Baseline Models）的相关代码和数据，包括`FastText`和`Random Forest`两种模型实现。
  - 包含的数据目录 `data/` 用于存放模型输入的训练、验证、测试数据，以及停用词表文件。
- **`BERT/`**
  - 用于实现基于BERT模型的分类任务，包含BERT相关的预训练模型、数据、配置、模型定义、训练代码等。
- **`README.md`**
  - 项目说明文档，用于概述项目目标、使用说明和技术细节。
- **`requirements.txt`**
  - 列出项目所需的依赖库，可通过 `pip install -r requirements.txt` 安装。

### 4.2 baseline 目录

1. `data/`

   - `class.txt`：定义类别标签文件。

   - `train.txt`, `dev.txt`, `test.txt`：分别为训练集、验证集和测试集（文本格式）。

   - `\*_fast.txt`：与上述数据集功能相同，但是为了快速处理设计的版本。

   - `\*_new.csv`：与上述数据集对应的CSV格式文件。

   - `stopwords.txt`：中文停用词表，用于预处理文本时剔除常见无意义词汇。

2. `fasttext/`

   - `app.py`：FastText模型的服务端实现，用于部署和提供API服务。

   - `fast_text_baseline1/2/3.ipynb`：不同实验版本的FastText基线模型代码，用于分析和比较实验结果。

   - `preprocess.ipynb` 和 `preprocess1.ipynb`：数据预处理代码，包括文本分词、特征提取等步骤。

   - `test.py`：FastText模型测试代码。

   - `toutiao_fasttext_\*.bin`：训练好的FastText模型文件，存储在二进制格式中，供后续加载和推断使用。

3. `random_forest/`

   - `analysis.ipynb`：随机森林模型的分析实验代码。

   - `random_forest_baseline1.ipynb`：基于随机森林模型的基线实验代码，用于实现分类任务。

------

### 4.3 BERT 目录

1. `bert_pretrain/`

   - **`bert_config.json`**：BERT预训练模型的配置文件，定义了模型的结构和超参数。

   - **`pytorch_model.bin`**：BERT预训练模型的权重文件，基于`transformers`库加载。

   - **`vocab.txt`**：BERT模型使用的词表，定义了可用的词汇和其对应的索引。

2. `cache/`
   - 用于存放模型加载过程中产生的缓存文件。

3. `config/`

   - `__init__.py`：配置模块的初始化文件。

   - **`config.py`**：定义项目配置类（`Config`），包括超参数、数据路径、BERT路径等设置。

4. `data/`

   - **`class.txt`**：定义类别标签文件。

   - **`train.txt`, `dev.txt`, `test.txt`**：训练集、验证集和测试集。

   - **`stopwords.txt`**：中文停用词表。

5. `model/`

   - `__init__.py`：模型模块初始化文件。

   - **`bert_classification.py`**：定义基于BERT的分类模型。

   - **`train_eval.py`**：包含训练和评估代码的模块，用于训练和验证BERT模型。

6. `utils/`

   - `__init__.py`：工具模块初始化文件。

   - **`data_helpers.py`**：包含构建词表、数据加载、预处理等辅助函数。

7. 其他文件

   - **`bert_classification.ipynb`**：BERT分类模型的实验记录，用于记录和分析实验结果。

   - **`run.py`**：项目的主运行脚本，负责加载配置、数据、模型，并启动训练和评估。

------

### 4.4 项目功能概述

1. **基线模型（Baseline Models）**
   - 实现了 `FastText` 和 `Random Forest` 两种分类方法。
   - 提供了多版本的实验记录和分析代码，便于比较不同实验结果。
2. **BERT模型**
   - 使用预训练的`BERT-base-chinese`模型，结合分类头实现中文文本分类任务。
   - 支持模型训练、验证和测试，并包含多个模块化的配置和工具代码。
3. **服务端部署**
   - 提供了 `FastText` 模型的服务端代码（`app.py`），可以通过API进行分类推断。

## 5. 使用方法

### 5.1 随机森林

随机森林基线模型只需要运行`random_forest_baseline1.ipynb`文件即可，运行结果如下：

```css
数据分割完毕，开始模型训练...
模型训练结束，开始预测...
ic| accuracy: 0.8001333333333334
```

### 5.2 FastText

#### 5.2.1 FastText_baseline1

FastText基线模型1对原始数据进行最基础的按字切分处理，将处理完的数据直接用`fasttext`进行训练，设置`wordNgrams=2`，直接运行`fast_text_baseline1.ipynb`文件，运行结果如下：

```css
Read 3M words
Number of words:  4760
Number of labels: 10
Progress:  96.4% words/sec/thread:  127822 lr:  0.003614 avg.loss:  0.398935 ETA:   0h 0m 0s
(10000, 0.9157, 0.9157)
Progress: 100.0% words/sec/thread:  121702 lr:  0.000000 avg.loss:  0.391975 ETA:   0h 0m 0s
```

#### 5.2.2 FastText_baseline2

FastText基线模型2在模型1的基础上设置更多的参数，设置`wordNgrams=2`以及`autotuneDuration=600`，运行10分钟搜索最佳的参数组合，直接运行`fast_text_baseline2.ipynb`文件，运行结果如下：

```css
Warning : wordNgrams is manually set to a specific value. It will not be automatically optimized.
Training again with best arguments
Read 3M words
Number of words:  4760
Number of labels: 10
Best selected args = 0
epoch = 100
lr = 0.160709
dim = 161
minCount = 1
wordNgrams = 2
minn = 0
maxn = 0
bucket = 10000000
dsub = 8
loss = softmax
Progress: 100.0% words/sec/thread:   49980 lr:  0.000000 avg.loss:  0.016898 ETA:   0h 0m 0s  50277 lr:  0.075192 avg.loss:  0.029494 ETA:   0h 0m55s 0.018784 ETA:   0h 0m13s
(10000, 0.9179, 0.9179)
```

#### 5.2.3 FastText_baseline3

FastText基线模型3和前两个模型在数据预处理上有所不同，基线模型三对原始数据进行分词处理处理，将处理完的数据直接用`fasttext`进行训练，设置`wordNgrams=2`以及`autotuneDuration=600`，运行10分钟搜索最佳的参数组合，直接运行`fast_text_baseline3.ipynb`文件，运行结果如下：

```css
Warning : wordNgrams is manually set to a specific value. It will not be automatically optimized.
Training again with best arguments
Best selected args = 0
epoch = 17
lr = 0.0439439
dim = 62
minCount = 1
wordNgrams = 2
minn = 2
maxn = 5
bucket = 1809723
dsub = 2
loss = softmax
Read 2M words
Number of words:  118456
Number of labels: 10
Progress:  99.4% words/sec/thread:   43137 lr:  0.000249 avg.loss:  0.298942 ETA:   0h 0m 0s
(10000, 0.9204, 0.9204)
Progress: 100.0% words/sec/thread:   43054 lr:  0.000000 avg.loss:  0.297841 ETA:   0h 0m 0s
```

### 5.3 BERT-Classification

#### 5.3.1 BERT配置

`bert_pretrain`文件夹下的`config.json`配置文件如下：

```json
{
  "attention_probs_dropout_prob": 0.1, 
  "directionality": "bidi", 
  "hidden_act": "gelu", 
  "hidden_dropout_prob": 0.1, 
  "hidden_size": 768, 
  "initializer_range": 0.02, 
  "intermediate_size": 3072, 
  "max_position_embeddings": 512, 
  "num_attention_heads": 12, 
  "num_hidden_layers": 12, 
  "pooler_fc_size": 768, 
  "pooler_num_attention_heads": 12, 
  "pooler_num_fc_layers": 3, 
  "pooler_size_per_head": 128, 
  "pooler_type": "first_token_transform", 
  "type_vocab_size": 2, 
  "vocab_size": 21128
}
```

模型架构：

```css
Model(
  (bert): BertModel(
    (embeddings): BertEmbeddings(
      (word_embeddings): Embedding(21128, 768, padding_idx=0)
      (position_embeddings): Embedding(512, 768)
      (token_type_embeddings): Embedding(2, 768)
      (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (encoder): BertEncoder(
      (layer): ModuleList(
        (0-11): 12 x BertLayer(
          (attention): BertAttention(
            (self): BertSdpaSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation()
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
    )
    (pooler): BertPooler(
      (dense): Linear(in_features=768, out_features=768, bias=True)
      (activation): Tanh()
    )
  )
  (fc): Linear(in_features=768, out_features=10, bias=True)
)
```

#### 5.3.2 训练和预测

直接执行如下命令即可进行模型训练和预测

```shell
python run_bert.py --model bert
```

训练过程和结果：

```css
Loading data for Bert Model...
180000it [00:31, 5804.22it/s]
10000it [00:01, 6143.86it/s]
10000it [00:01, 6146.91it/s]
Epoch [1/5]
 14%|██████████████████████▌                                                                                                                                        | 200/1407 [01:36<09:45,  2.06it/s]Iter: 200, Train Loss: 0.38, Train Acc: 89.06%, Val Loss: 0.32, Val Acc: 90.48%, Time: 0:01:51 *
 28%|█████████████████████████████████████████████▏                                                                                                                 | 400/1407 [03:25<07:51,  2.14it/s]Iter: 400, Train Loss: 0.39, Train Acc: 89.84%, Val Loss: 0.28, Val Acc: 91.47%, Time: 0:03:39 *
 43%|███████████████████████████████████████████████████████████████████▊                                                                                           | 600/1407 [05:14<06:12,  2.17it/s]Iter: 600, Train Loss: 0.31, Train Acc: 91.41%, Val Loss: 0.25, Val Acc: 92.22%, Time: 0:05:27 *
 57%|██████████████████████████████████████████████████████████████████████████████████████████▍                                                                    | 800/1407 [07:01<04:40,  2.16it/s]Iter: 800, Train Loss: 0.16, Train Acc: 95.31%, Val Loss: 0.23, Val Acc: 92.80%, Time: 0:07:15 *
 71%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                             | 1000/1407 [08:49<03:11,  2.12it/s]Iter: 1000, Train Loss: 0.21, Train Acc: 95.31%, Val Loss: 0.24, Val Acc: 92.63%, Time: 0:09:02 
 85%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                       | 1200/1407 [10:37<01:37,  2.12it/s]Iter: 1200, Train Loss: 0.20, Train Acc: 92.97%, Val Loss: 0.21, Val Acc: 92.68%, Time: 0:10:52 *
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏| 1400/1407 [12:27<00:03,  2.15it/s]Iter: 1400, Train Loss: 0.37, Train Acc: 91.41%, Val Loss: 0.21, Val Acc: 93.41%, Time: 0:12:41 *
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1407/1407 [12:43<00:00,  1.84it/s]
Epoch [2/5]
 14%|█████████████████████▊                                                                                                                                         | 193/1407 [01:32<09:17,  2.18it/s]Iter: 1600, Train Loss: 0.31, Train Acc: 91.41%, Val Loss: 0.21, Val Acc: 93.18%, Time: 0:14:28 
 28%|████████████████████████████████████████████▍                                                                                                                  | 393/1407 [03:20<07:46,  2.17it/s]Iter: 1800, Train Loss: 0.20, Train Acc: 95.31%, Val Loss: 0.21, Val Acc: 93.41%, Time: 0:16:16 *
 42%|███████████████████████████████████████████████████████████████████                                                                                            | 593/1407 [05:08<06:14,  2.18it/s]Iter: 2000, Train Loss: 0.16, Train Acc: 95.31%, Val Loss: 0.21, Val Acc: 93.62%, Time: 0:18:05 *
 52%|██████████████████████████████████████████████████████████████████████████████████                                                                             | 726/1407 [06:24<05:2 56%|█████████████████████████████████████████████████████████████████████████████████████████▌                                                                     | 793/1407 [06:56<04:45,  2.15it/s]Iter: 2200, Train Loss: 0.17, Train Acc: 95.31%, Val Loss: 0.21, Val Acc: 93.65%, Time: 0:19:53 
 71%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                              | 993/1407 [08:44<03:11,  2.16it/s]Iter: 2400, Train Loss: 0.08, Train Acc: 97.66%, Val Loss: 0.20, Val Acc: 93.48%, Time: 0:21:41 *
 85%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                        | 1193/1407 [10:32<01:39,  2.16it/s]Iter: 2600, Train Loss: 0.14, Train Acc: 95.31%, Val Loss: 0.20, Val Acc: 93.57%, Time: 0:23:28 *
 99%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍ | 1393/1407 [12:20<00:06,  2.15it/s]Iter: 2800, Train Loss: 0.09, Train Acc: 97.66%, Val Loss: 0.20, Val Acc: 93.38%, Time: 0:25:16 
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1407/1407 [12:39<00:00,  1.85it/s]
Epoch [3/5]
 13%|█████████████████████                                                                                                                                          | 186/1407 [01:28<09:28,  2.15it/s]Iter: 3000, Train Loss: 0.12, Train Acc: 96.09%, Val Loss: 0.21, Val Acc: 93.51%, Time: 0:27:04 
 27%|███████████████████████████████████████████▌                                                                                                                   | 386/1407 [03:16<07:55,  2.15it/s]Iter: 3200, Train Loss: 0.23, Train Acc: 95.31%, Val Loss: 0.21, Val Acc: 93.54%, Time: 0:28:52 
 42%|██████████████████████████████████████████████████████████████████▏                                                                                            | 586/1407 [05:04<06:20,  2.16it/s]Iter: 3400, Train Loss: 0.16, Train Acc: 94.53%, Val Loss: 0.21, Val Acc: 93.73%, Time: 0:30:40 
 56%|████████████████████████████████████████████████████████████████████████████████████████▊                                                                      | 786/1407 [06:52<04:47,  2.16it/s]Iter: 3600, Train Loss: 0.05, Train Acc: 97.66%, Val Loss: 0.22, Val Acc: 93.49%, Time: 0:32:28 
No optimization for a long time, auto-stopping...
 56%|████████████████████████████████████████████████████████████████████████████████████████▊                                                                      | 786/1407 [07:05<05:36,  1.85it/s]
```

预测结果：

```shell
Test Loss:  0.21, Test Acc: 94.05%
Precision, Recall and F1-Score...
               precision    recall  f1-score   support

      finance     0.9043    0.9450    0.9242      1000
       realty     0.9440    0.9600    0.9519      1000
       stocks     0.9113    0.8840    0.8975      1000
    education     0.9661    0.9680    0.9670      1000
      science     0.9118    0.9100    0.9109      1000
      society     0.9394    0.9300    0.9347      1000
     politics     0.9392    0.9110    0.9249      1000
       sports     0.9820    0.9820    0.9820      1000
         game     0.9645    0.9510    0.9577      1000
entertainment     0.9432    0.9640    0.9535      1000

     accuracy                         0.9405     10000
    macro avg     0.9406    0.9405    0.9404     10000
 weighted avg     0.9406    0.9405    0.9404     10000

Confusion Matrix...
[[945  11  30   1   5   1   5   1   0   1]
 [ 10 960  11   2   4   4   2   2   0   5]
 [ 52  15 884   0  17   2  29   0   0   1]
 [  2   1   2 968   4   6   5   0   2  10]
 [  6   3  21   4 910  14   8   0  22  12]
 [ 12  13   2  15   3 930   9   0   5  11]
 [ 10   6  16  10  15  26 911   1   1   4]
 [  3   4   2   0   1   2   0 982   1   5]
 [  2   2   2   0  31   1   1   1 951   9]
 [  3   2   0   2   8   4   0  13   4 964]]
Time usage: 0:00:11
```

```css
Loading data for Bert Model...
180000it [00:30, 5999.82it/s]
10000it [00:01, 6217.58it/s]
10000it [00:01, 5889.53it/s]
Epoch [1/5]
 14%|██████████████████████▌                                                                                                                                        | 200/1407 [00:43<04:21,  4.62it/s]Iter: 200, Train Loss: 0.38, Train Acc: 89.06%, Val Loss: 0.32, Val Acc: 90.48%, Time: 0:00:50 *
 28%|█████████████████████████████████████████████▏                                                                                                                 | 400/1407 [01:33<03:42,  4.53it/s]Iter: 400, Train Loss: 0.39, Train Acc: 89.84%, Val Loss: 0.28, Val Acc: 91.47%, Time: 0:01:40 *
 43%|███████████████████████████████████████████████████████████████████▊                                                                                           | 600/1407 [02:24<03:00,  4.48it/s]Iter: 600, Train Loss: 0.31, Train Acc: 91.41%, Val Loss: 0.25, Val Acc: 92.22%, Time: 0:02:32 *
 57%|██████████████████████████████████████████████████████████████████████████████████████████▍                                                                    | 800/1407 [03:15<02:14,  4.51it/s]Iter: 800, Train Loss: 0.16, Train Acc: 95.31%, Val Loss: 0.23, Val Acc: 92.80%, Time: 0:03:23 *
 71%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                             | 1000/1407 [04:07<01:30,  4.48it/s]Iter: 1000, Train Loss: 0.21, Train Acc: 95.31%, Val Loss: 0.24, Val Acc: 92.63%, Time: 0:04:13 
 85%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                       | 1200/1407 [04:57<00:45,  4.50it/s]Iter: 1200, Train Loss: 0.20, Train Acc: 92.97%, Val Loss: 0.21, Val Acc: 92.68%, Time: 0:05:04 *
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏| 1400/1407 [05:48<00:01,  4.48it/s]Iter: 1400, Train Loss: 0.37, Train Acc: 91.41%, Val Loss: 0.21, Val Acc: 93.41%, Time: 0:05:55 *
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1407/1407 [05:55<00:00,  3.95it/s]
Epoch [2/5]
 14%|█████████████████████▊                                                                                                                                         | 193/1407 [00:42<04:30,  4.49it/s]Iter: 1600, Train Loss: 0.31, Train Acc: 91.41%, Val Loss: 0.21, Val Acc: 93.18%, Time: 0:06:45 
 28%|████████████████████████████████████████████▍                                                                                                                  | 393/1407 [01:32<03:45,  4.49it/s]Iter: 1800, Train Loss: 0.20, Train Acc: 95.31%, Val Loss: 0.21, Val Acc: 93.41%, Time: 0:07:36 *
 42%|███████████████████████████████████████████████████████████████████                                                                                            | 593/1407 [02:23<03:00,  4.51it/s]Iter: 2000, Train Loss: 0.16, Train Acc: 95.31%, Val Loss: 0.21, Val Acc: 93.62%, Time: 0:08:27 *
 56%|█████████████████████████████████████████████████████████████████████████████████████████▌                                                                     | 793/1407 [03:14<02:17,  4.46it/s]Iter: 2200, Train Loss: 0.17, Train Acc: 95.31%, Val Loss: 0.21, Val Acc: 93.65%, Time: 0:09:17 
 71%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                              | 993/1407 [04:04<01:32,  4.49it/s]Iter: 2400, Train Loss: 0.08, Train Acc: 97.66%, Val Loss: 0.20, Val Acc: 93.48%, Time: 0:10:07 *
 85%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                        | 1193/1407 [04:55<00:47,  4.48it/s]Iter: 2600, Train Loss: 0.14, Train Acc: 95.31%, Val Loss: 0.20, Val Acc: 93.57%, Time: 0:10:59 *
 99%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍ | 1393/1407 [05:46<00:03,  4.48it/s]Iter: 2800, Train Loss: 0.09, Train Acc: 97.66%, Val Loss: 0.20, Val Acc: 93.38%, Time: 0:11:49 
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1407/1407 [05:55<00:00,  3.96it/s]
Epoch [3/5]
 13%|█████████████████████                                                                                                                                          | 186/1407 [00:41<04:30,  4.51it/s]Iter: 3000, Train Loss: 0.12, Train Acc: 96.09%, Val Loss: 0.21, Val Acc: 93.51%, Time: 0:12:39 
 27%|███████████████████████████████████████████▌                                                                                                                   | 386/1407 [01:31<03:46,  4.51it/s]Iter: 3200, Train Loss: 0.23, Train Acc: 95.31%, Val Loss: 0.21, Val Acc: 93.54%, Time: 0:13:29 
 42%|██████████████████████████████████████████████████████████████████▏                                                                                            | 586/1407 [02:21<03:02,  4.50it/s]Iter: 3400, Train Loss: 0.16, Train Acc: 94.53%, Val Loss: 0.21, Val Acc: 93.73%, Time: 0:14:19 
 56%|████████████████████████████████████████████████████████████████████████████████████████▊                                                                      | 786/1407 [03:11<02:18,  4.48it/s]Iter: 3600, Train Loss: 0.05, Train Acc: 97.66%, Val Loss: 0.22, Val Acc: 93.49%, Time: 0:15:09 
No optimization for a long time, auto-stopping...
 56%|████████████████████████████████████████████████████████████████████████████████████████▊                                                                      | 786/1407 [03:17<02:35,  3.98it/s]
Test Loss:  0.21, Test Acc: 94.05%
Precision, Recall and F1-Score...
               precision    recall  f1-score   support

      finance     0.9043    0.9450    0.9242      1000
       realty     0.9440    0.9600    0.9519      1000
       stocks     0.9113    0.8840    0.8975      1000
    education     0.9661    0.9680    0.9670      1000
      science     0.9118    0.9100    0.9109      1000
      society     0.9394    0.9300    0.9347      1000
     politics     0.9392    0.9110    0.9249      1000
       sports     0.9820    0.9820    0.9820      1000
         game     0.9645    0.9510    0.9577      1000
entertainment     0.9432    0.9640    0.9535      1000

     accuracy                         0.9405     10000
    macro avg     0.9406    0.9405    0.9404     10000
 weighted avg     0.9406    0.9405    0.9404     10000

Confusion Matrix...
[[945  11  30   1   5   1   5   1   0   1]
 [ 10 960  11   2   4   4   2   2   0   5]
 [ 52  15 884   0  17   2  29   0   0   1]
 [  2   1   2 968   4   6   5   0   2  10]
 [  6   3  21   4 910  14   8   0  22  12]
 [ 12  13   2  15   3 930   9   0   5  11]
 [ 10   6  16  10  15  26 911   1   1   4]
 [  3   4   2   0   1   2   0 982   1   5]
 [  2   2   2   0  31   1   1   1 951   9]
 [  3   2   0   2   8   4   0  13   4 964]]
Time usage: 0:00:06
```

#### 5.3.3 模型量化

直接执行如下命令即可进行模型的量化：

```shell
python run_bert_quantize.py --model bert
```

剪枝后的模型架构：

```css
Model(
  (bert): BertModel(
    (embeddings): BertEmbeddings(
      (word_embeddings): Embedding(21128, 768, padding_idx=0)
      (position_embeddings): Embedding(512, 768)
      (token_type_embeddings): Embedding(2, 768)
      (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (encoder): BertEncoder(
      (layer): ModuleList(
        (0-11): 12 x BertLayer(
          (attention): BertAttention(
            (self): BertSdpaSelfAttention(
              (query): DynamicQuantizedLinear(in_features=768, out_features=768, dtype=torch.qint8, qscheme=torch.per_tensor_affine)
              (key): DynamicQuantizedLinear(in_features=768, out_features=768, dtype=torch.qint8, qscheme=torch.per_tensor_affine)
              (value): DynamicQuantizedLinear(in_features=768, out_features=768, dtype=torch.qint8, qscheme=torch.per_tensor_affine)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (output): BertSelfOutput(
              (dense): DynamicQuantizedLinear(in_features=768, out_features=768, dtype=torch.qint8, qscheme=torch.per_tensor_affine)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (intermediate): BertIntermediate(
            (dense): DynamicQuantizedLinear(in_features=768, out_features=3072, dtype=torch.qint8, qscheme=torch.per_tensor_affine)
            (intermediate_act_fn): GELUActivation()
          )
          (output): BertOutput(
            (dense): DynamicQuantizedLinear(in_features=3072, out_features=768, dtype=torch.qint8, qscheme=torch.per_tensor_affine)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
    )
    (pooler): BertPooler(
      (dense): DynamicQuantizedLinear(in_features=768, out_features=768, dtype=torch.qint8, qscheme=torch.per_tensor_affine)
      (activation): Tanh()
    )
  )
  (fc): DynamicQuantizedLinear(in_features=768, out_features=10, dtype=torch.qint8, qscheme=torch.per_tensor_affine)
)
```

剪枝后的在测试集上的预测结果如下：

```css
Test Loss:   0.2, Test Acc: 93.81%
Precision, Recall and F1-Score...
               precision    recall  f1-score   support

      finance     0.9474    0.9010    0.9236      1000
       realty     0.9387    0.9640    0.9512      1000
       stocks     0.8959    0.8780    0.8869      1000
    education     0.9649    0.9610    0.9629      1000
      science     0.8807    0.9230    0.9014      1000
      society     0.9384    0.9290    0.9337      1000
     politics     0.9281    0.9290    0.9285      1000
       sports     0.9898    0.9750    0.9824      1000
         game     0.9714    0.9500    0.9606      1000
entertainment     0.9301    0.9710    0.9501      1000

     accuracy                         0.9381     10000
    macro avg     0.9385    0.9381    0.9381     10000
 weighted avg     0.9385    0.9381    0.9381     10000

Confusion Matrix...
[[901  16  54   1  10   6   8   1   0   3]
 [  5 964   8   2   3   3   4   1   0  10]
 [ 34  18 878   2  41   2  21   1   0   3]
 [  3   2   0 961   5  12   5   0   0  12]
 [  1   3  14   2 923  11  11   0  22  13]
 [  4  10   2  13  10 929  17   0   2  13]
 [  2   7  18  11  14  16 929   0   0   3]
 [  1   3   3   0   2   5   2 975   0   9]
 [  0   1   3   1  32   3   3   0 950   7]
 [  0   3   0   3   8   3   1   7   4 971]]
Time usage: 0:01:24
```

剪枝前后对比：

```shell
-rw-r--r-- 1 root root 409169465 12月 30 16:44 bert.pt	390MB
-rw-r--r-- 1 root root 152656502 12月 30 17:03 bert_quantize.pt	145MB
```

剪枝后的模型大小仅需145MB，精度仅仅下降百分0.2，模型的效果非常优异。

### 5.4 ALBERT-Classification

#### 5.4.1 ALBERT配置

`albert_chinese_base`配置如下：

```json
{
  "architectures": [
    "AlbertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0,
  "bos_token_id": 2,
  "classifier_dropout_prob": 0.1,
  "down_scale_factor": 1,
  "embedding_size": 128,
  "eos_token_id": 3,
  "gap_size": 0,
  "hidden_act": "relu",
  "hidden_dropout_prob": 0,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "inner_group_num": 1,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "layers_to_keep": [],
  "max_position_embeddings": 512,
  "model_type": "albert",
  "net_structure_type": 0,
  "num_attention_heads": 12,
  "num_hidden_groups": 1,
  "num_hidden_layers": 12,
  "num_memory_blocks": 0,
  "pad_token_id": 0,
  "type_vocab_size": 2,
  "vocab_size": 21128
}
```

ALBERT模型架构如下：

```css
Model(
  (albert): AlbertModel(
    (embeddings): AlbertEmbeddings(
      (word_embeddings): Embedding(21128, 128, padding_idx=0)
      (position_embeddings): Embedding(512, 128)
      (token_type_embeddings): Embedding(2, 128)
      (LayerNorm): LayerNorm((128,), eps=1e-12, elementwise_affine=True)
      (dropout): Dropout(p=0, inplace=False)
    )
    (encoder): AlbertTransformer(
      (embedding_hidden_mapping_in): Linear(in_features=128, out_features=768, bias=True)
      (albert_layer_groups): ModuleList(
        (0): AlbertLayerGroup(
          (albert_layers): ModuleList(
            (0): AlbertLayer(
              (full_layer_layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (attention): AlbertSdpaAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (attention_dropout): Dropout(p=0, inplace=False)
                (output_dropout): Dropout(p=0, inplace=False)
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              )
              (ffn): Linear(in_features=768, out_features=3072, bias=True)
              (ffn_output): Linear(in_features=3072, out_features=768, bias=True)
              (activation): ReLU()
              (dropout): Dropout(p=0, inplace=False)
            )
          )
        )
      )
    )
    (pooler): Linear(in_features=768, out_features=768, bias=True)
    (pooler_activation): Tanh()
  )
  (fc): Linear(in_features=768, out_features=10, bias=True)
)
```

#### 5.4.2 训练和预测

直接执行如下命令进行训练:

```shell
python run_albert.py --model albert
```

训练过程和结果：

```css
Loading data for Bert Model...
180000it [00:30, 5995.86it/s]
10000it [00:01, 6333.72it/s]
10000it [00:01, 6289.69it/s]
Epoch [1/5]
 14%|██████████████████████▌                                                                                                                                        | 200/1407 [00:39<03:55,  5.13it/s]Iter: 200, Train Loss: 1.20, Train Acc: 63.28%, Val Loss: 1.08, Val Acc: 65.37%, Time: 0:00:45 *
 28%|█████████████████████████████████████████████▏                                                                                                                 | 400/1407 [01:24<03:21,  5.01it/s]Iter: 400, Train Loss: 0.79, Train Acc: 78.91%, Val Loss: 0.76, Val Acc: 76.43%, Time: 0:01:30 *
 43%|███████████████████████████████████████████████████████████████████▊                                                                                           | 600/1407 [02:10<02:44,  4.90it/s]Iter: 600, Train Loss: 0.56, Train Acc: 84.38%, Val Loss: 0.56, Val Acc: 82.13%, Time: 0:02:16 *
 57%|██████████████████████████████████████████████████████████████████████████████████████████▍                                                                    | 800/1407 [02:56<02:01,  4.99it/s]Iter: 800, Train Loss: 0.48, Train Acc: 85.16%, Val Loss: 0.49, Val Acc: 85.10%, Time: 0:03:02 *
 71%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                             | 1000/1407 [03:41<01:21,  4.98it/s]Iter: 1000, Train Loss: 0.35, Train Acc: 85.94%, Val Loss: 0.47, Val Acc: 85.62%, Time: 0:03:48 *
 85%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                       | 1200/1407 [04:27<00:41,  4.97it/s]Iter: 1200, Train Loss: 0.38, Train Acc: 89.84%, Val Loss: 0.42, Val Acc: 87.24%, Time: 0:04:34 *
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏| 1400/1407 [05:13<00:01,  4.96it/s]Iter: 1400, Train Loss: 0.45, Train Acc: 86.72%, Val Loss: 0.41, Val Acc: 87.63%, Time: 0:05:20 *
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1407/1407 [05:20<00:00,  4.39it/s]
Epoch [2/5]
 14%|█████████████████████▊                                                                                                                                         | 193/1407 [00:38<04:03,  4.98it/s]Iter: 1600, Train Loss: 0.41, Train Acc: 85.94%, Val Loss: 0.41, Val Acc: 87.44%, Time: 0:06:05 
 28%|████████████████████████████████████████████▍                                                                                                                  | 393/1407 [01:24<03:24,  4.96it/s]Iter: 1800, Train Loss: 0.29, Train Acc: 92.19%, Val Loss: 0.37, Val Acc: 88.64%, Time: 0:06:51 *
 42%|███████████████████████████████████████████████████████████████████                                                                                            | 593/1407 [02:10<02:43,  4.99it/s]Iter: 2000, Train Loss: 0.45, Train Acc: 85.94%, Val Loss: 0.37, Val Acc: 88.46%, Time: 0:07:37 *
 56%|█████████████████████████████████████████████████████████████████████████████████████████▌                                                                     | 793/1407 [02:56<02:03,  4.96it/s]Iter: 2200, Train Loss: 0.26, Train Acc: 92.19%, Val Loss: 0.33, Val Acc: 89.66%, Time: 0:08:23 *
 71%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                              | 993/1407 [03:42<01:23,  4.96it/s]Iter: 2400, Train Loss: 0.33, Train Acc: 92.19%, Val Loss: 0.36, Val Acc: 89.33%, Time: 0:09:09 
 85%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                        | 1193/1407 [04:28<00:43,  4.96it/s]Iter: 2600, Train Loss: 0.34, Train Acc: 89.84%, Val Loss: 0.35, Val Acc: 89.00%, Time: 0:09:55 
 99%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍ | 1393/1407 [05:14<00:02,  4.95it/s]Iter: 2800, Train Loss: 0.31, Train Acc: 92.97%, Val Loss: 0.32, Val Acc: 90.16%, Time: 0:10:41 *
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1407/1407 [05:22<00:00,  4.36it/s]
Epoch [3/5]
 13%|█████████████████████                                                                                                                                          | 186/1407 [00:37<04:06,  4.96it/s]Iter: 3000, Train Loss: 0.29, Train Acc: 91.41%, Val Loss: 0.35, Val Acc: 89.60%, Time: 0:11:26 
 27%|███████████████████████████████████████████▌                                                                                                                   | 386/1407 [01:23<03:26,  4.96it/s]Iter: 3200, Train Loss: 0.50, Train Acc: 89.06%, Val Loss: 0.34, Val Acc: 89.79%, Time: 0:12:12 
 42%|██████████████████████████████████████████████████████████████████▏                                                                                            | 586/1407 [02:09<02:45,  4.95it/s]Iter: 3400, Train Loss: 0.41, Train Acc: 89.84%, Val Loss: 0.31, Val Acc: 90.32%, Time: 0:12:58 *
 56%|████████████████████████████████████████████████████████████████████████████████████████▊                                                                      | 786/1407 [02:54<02:05,  4.95it/s]Iter: 3600, Train Loss: 0.15, Train Acc: 96.88%, Val Loss: 0.31, Val Acc: 90.62%, Time: 0:13:44 *
 70%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                               | 986/1407 [03:40<01:24,  4.96it/s]Iter: 3800, Train Loss: 0.25, Train Acc: 89.06%, Val Loss: 0.35, Val Acc: 89.72%, Time: 0:14:30 
 84%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                        | 1186/1407 [04:26<00:44,  4.96it/s]Iter: 4000, Train Loss: 0.21, Train Acc: 94.53%, Val Loss: 0.32, Val Acc: 90.06%, Time: 0:15:15 
 99%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋  | 1386/1407 [05:12<00:04,  4.97it/s]Iter: 4200, Train Loss: 0.32, Train Acc: 91.41%, Val Loss: 0.31, Val Acc: 90.43%, Time: 0:16:01 
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1407/1407 [05:21<00:00,  4.37it/s]
Epoch [4/5]
 13%|████████████████████▏                                                                                                                                          | 179/1407 [00:35<04:07,  4.96it/s]Iter: 4400, Train Loss: 0.17, Train Acc: 97.66%, Val Loss: 0.30, Val Acc: 90.65%, Time: 0:16:47 *
 27%|██████████████████████████████████████████▊                                                                                                                    | 379/1407 [01:21<03:27,  4.96it/s]Iter: 4600, Train Loss: 0.21, Train Acc: 92.97%, Val Loss: 0.29, Val Acc: 91.04%, Time: 0:17:33 *
 41%|█████████████████████████████████████████████████████████████████▍                                                                                             | 579/1407 [02:07<02:46,  4.96it/s]Iter: 4800, Train Loss: 0.10, Train Acc: 97.66%, Val Loss: 0.29, Val Acc: 91.20%, Time: 0:18:19 
 55%|████████████████████████████████████████████████████████████████████████████████████████                                                                       | 779/1407 [02:53<02:06,  4.95it/s]Iter: 5000, Train Loss: 0.18, Train Acc: 94.53%, Val Loss: 0.31, Val Acc: 90.54%, Time: 0:19:04 
 70%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                                | 979/1407 [03:39<01:26,  4.97it/s]Iter: 5200, Train Loss: 0.26, Train Acc: 91.41%, Val Loss: 0.31, Val Acc: 90.93%, Time: 0:19:50 
 84%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                         | 1179/1407 [04:25<00:45,  4.96it/s]Iter: 5400, Train Loss: 0.27, Train Acc: 89.84%, Val Loss: 0.30, Val Acc: 91.19%, Time: 0:20:36 
 98%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊   | 1379/1407 [05:10<00:05,  4.96it/s]Iter: 5600, Train Loss: 0.12, Train Acc: 96.09%, Val Loss: 0.29, Val Acc: 90.99%, Time: 0:21:22 
No optimization for a long time, auto-stopping...
 98%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊   | 1379/1407 [05:16<00:06,  4.36it/s]
```

预测结果如下:

```css
Test Loss:  0.29, Test Acc: 91.34%
Precision, Recall and F1-Score...
               precision    recall  f1-score   support

      finance     0.8754    0.9200    0.8971      1000
       realty     0.9200    0.9310    0.9254      1000
       stocks     0.8891    0.8020    0.8433      1000
    education     0.9554    0.9640    0.9597      1000
      science     0.8423    0.8920    0.8664      1000
      society     0.9174    0.9000    0.9086      1000
     politics     0.9303    0.8670    0.8975      1000
       sports     0.9896    0.9480    0.9683      1000
         game     0.9366    0.9460    0.9413      1000
entertainment     0.8877    0.9640    0.9243      1000

     accuracy                         0.9134     10000
    macro avg     0.9144    0.9134    0.9132     10000
 weighted avg     0.9144    0.9134    0.9132     10000

Confusion Matrix...
[[920  15  33   2  15   2   5   1   2   5]
 [ 12 931  11   1  10   8   6   1   2  18]
 [ 86  27 802   1  55   2  17   1   5   4]
 [  3   0   0 964   5  12   5   0   1  10]
 [  6   4  17   7 892  15   8   0  34  17]
 [  9  21   3  20  14 900  14   0   2  17]
 [ 12   7  31   8  23  30 867   2   6  14]
 [  1   4   1   1   2   4   3 948   3  33]
 [  0   1   3   1  34   7   3   1 946   4]
 [  2   2   1   4   9   1   4   4   9 964]]
Time usage: 0:00:05
```

### 5.5 T5-Finetune

#### 5.5.1 T5配置

`t5-base-chinese`配置如下：

```json
{
  "architectures": [
    "T5ForConditionalGeneration"
  ],
  "d_ff": 2048,
  "d_kv": 64,
  "d_model": 768,
  "decoder_start_token_id": 0,
  "dropout_rate": 0.1,
  "eos_token_id": 1,
  "feed_forward_proj": "gated-gelu",
  "initializer_factor": 1.0,
  "is_encoder_decoder": true,
  "layer_norm_epsilon": 1e-06,
  "model_type": "mt5",
  "num_decoder_layers": 12,
  "num_heads": 12,
  "num_layers": 12,
  "output_past": true,
  "pad_token_id": 0,
  "relative_attention_num_buckets": 32,
  "tie_word_embeddings": false,
  "tokenizer_class": "T5Tokenizer",
  "vocab_size": 50000
}
```

T5模型架构如下:

```css
Model(
  (t5): T5Model(
    (shared): Embedding(50000, 768)
    (encoder): T5Stack(
      (embed_tokens): Embedding(50000, 768)
      (block): ModuleList(
        (0): T5Block(
          (layer): ModuleList(
            (0): T5LayerSelfAttention(
              (SelfAttention): T5Attention(
                (q): Linear(in_features=768, out_features=768, bias=False)
                (k): Linear(in_features=768, out_features=768, bias=False)
                (v): Linear(in_features=768, out_features=768, bias=False)
                (o): Linear(in_features=768, out_features=768, bias=False)
                (relative_attention_bias): Embedding(32, 12)
              )
              (layer_norm): FusedRMSNorm(torch.Size([768]), eps=1e-06, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (1): T5LayerFF(
              (DenseReluDense): T5DenseGatedActDense(
                (wi_0): Linear(in_features=768, out_features=2048, bias=False)
                (wi_1): Linear(in_features=768, out_features=2048, bias=False)
                (wo): Linear(in_features=2048, out_features=768, bias=False)
                (dropout): Dropout(p=0.1, inplace=False)
                (act): NewGELUActivation()
              )
              (layer_norm): FusedRMSNorm(torch.Size([768]), eps=1e-06, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
        )
        (1-11): 11 x T5Block(
          (layer): ModuleList(
            (0): T5LayerSelfAttention(
              (SelfAttention): T5Attention(
                (q): Linear(in_features=768, out_features=768, bias=False)
                (k): Linear(in_features=768, out_features=768, bias=False)
                (v): Linear(in_features=768, out_features=768, bias=False)
                (o): Linear(in_features=768, out_features=768, bias=False)
              )
              (layer_norm): FusedRMSNorm(torch.Size([768]), eps=1e-06, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (1): T5LayerFF(
              (DenseReluDense): T5DenseGatedActDense(
                (wi_0): Linear(in_features=768, out_features=2048, bias=False)
                (wi_1): Linear(in_features=768, out_features=2048, bias=False)
                (wo): Linear(in_features=2048, out_features=768, bias=False)
                (dropout): Dropout(p=0.1, inplace=False)
                (act): NewGELUActivation()
              )
              (layer_norm): FusedRMSNorm(torch.Size([768]), eps=1e-06, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
        )
      )
      (final_layer_norm): FusedRMSNorm(torch.Size([768]), eps=1e-06, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (decoder): T5Stack(
      (embed_tokens): Embedding(50000, 768)
      (block): ModuleList(
        (0): T5Block(
          (layer): ModuleList(
            (0): T5LayerSelfAttention(
              (SelfAttention): T5Attention(
                (q): Linear(in_features=768, out_features=768, bias=False)
                (k): Linear(in_features=768, out_features=768, bias=False)
                (v): Linear(in_features=768, out_features=768, bias=False)
                (o): Linear(in_features=768, out_features=768, bias=False)
                (relative_attention_bias): Embedding(32, 12)
              )
              (layer_norm): FusedRMSNorm(torch.Size([768]), eps=1e-06, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (1): T5LayerCrossAttention(
              (EncDecAttention): T5Attention(
                (q): Linear(in_features=768, out_features=768, bias=False)
                (k): Linear(in_features=768, out_features=768, bias=False)
                (v): Linear(in_features=768, out_features=768, bias=False)
                (o): Linear(in_features=768, out_features=768, bias=False)
              )
              (layer_norm): FusedRMSNorm(torch.Size([768]), eps=1e-06, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (2): T5LayerFF(
              (DenseReluDense): T5DenseGatedActDense(
                (wi_0): Linear(in_features=768, out_features=2048, bias=False)
                (wi_1): Linear(in_features=768, out_features=2048, bias=False)
                (wo): Linear(in_features=2048, out_features=768, bias=False)
                (dropout): Dropout(p=0.1, inplace=False)
                (act): NewGELUActivation()
              )
              (layer_norm): FusedRMSNorm(torch.Size([768]), eps=1e-06, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
        )
        (1-11): 11 x T5Block(
          (layer): ModuleList(
            (0): T5LayerSelfAttention(
              (SelfAttention): T5Attention(
                (q): Linear(in_features=768, out_features=768, bias=False)
                (k): Linear(in_features=768, out_features=768, bias=False)
                (v): Linear(in_features=768, out_features=768, bias=False)
                (o): Linear(in_features=768, out_features=768, bias=False)
              )
              (layer_norm): FusedRMSNorm(torch.Size([768]), eps=1e-06, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (1): T5LayerCrossAttention(
              (EncDecAttention): T5Attention(
                (q): Linear(in_features=768, out_features=768, bias=False)
                (k): Linear(in_features=768, out_features=768, bias=False)
                (v): Linear(in_features=768, out_features=768, bias=False)
                (o): Linear(in_features=768, out_features=768, bias=False)
              )
              (layer_norm): FusedRMSNorm(torch.Size([768]), eps=1e-06, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (2): T5LayerFF(
              (DenseReluDense): T5DenseGatedActDense(
                (wi_0): Linear(in_features=768, out_features=2048, bias=False)
                (wi_1): Linear(in_features=768, out_features=2048, bias=False)
                (wo): Linear(in_features=2048, out_features=768, bias=False)
                (dropout): Dropout(p=0.1, inplace=False)
                (act): NewGELUActivation()
              )
              (layer_norm): FusedRMSNorm(torch.Size([768]), eps=1e-06, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
        )
      )
      (final_layer_norm): FusedRMSNorm(torch.Size([768]), eps=1e-06, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
  )
  (fc): Linear(in_features=768, out_features=10, bias=True)
```

#### 5.5.2 训练和预测

直接执行如下命令进行训练:

```shell
python run_t5.py --model t5
```

训练过程和结果：

```css
Loading data for t5 Model...
180000it [00:23, 7628.99it/s]
10000it [00:01, 7741.77it/s]
10000it [00:01, 7341.33it/s]
Epoch [1/5]
 14%|██████████████████████▌                                                                                                                                        | 200/1407 [00:55<05:30,  3.65it/s]Iter: 200, Train Loss: 2.30, Train Acc: 7.81%, Val Loss: 2.30, Val Acc: 12.00%, Time: 0:01:04 *
 28%|█████████████████████████████████████████████▏                                                                                                                 | 400/1407 [01:58<04:41,  3.57it/s]Iter: 400, Train Loss: 2.30, Train Acc: 14.06%, Val Loss: 2.30, Val Acc: 15.95%, Time: 0:02:08 *
 43%|███████████████████████████████████████████████████████████████████▊                                                                                           | 600/1407 [03:04<03:45,  3.58it/s]Iter: 600, Train Loss: 1.99, Train Acc: 33.59%, Val Loss: 2.00, Val Acc: 37.14%, Time: 0:03:14 *
 57%|██████████████████████████████████████████████████████████████████████████████████████████▍                                                                    | 800/1407 [04:09<02:50,  3.57it/s]Iter: 800, Train Loss: 0.65, Train Acc: 80.47%, Val Loss: 0.54, Val Acc: 84.13%, Time: 0:04:19 *
 71%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                             | 1000/1407 [05:14<01:53,  3.57it/s]Iter: 1000, Train Loss: 0.48, Train Acc: 83.59%, Val Loss: 0.46, Val Acc: 86.67%, Time: 0:05:24 *
 85%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                       | 1200/1407 [06:19<00:57,  3.58it/s]Iter: 1200, Train Loss: 0.40, Train Acc: 86.72%, Val Loss: 0.41, Val Acc: 87.79%, Time: 0:06:29 *
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏| 1400/1407 [07:24<00:01,  3.57it/s]Iter: 1400, Train Loss: 0.49, Train Acc: 85.16%, Val Loss: 0.39, Val Acc: 88.28%, Time: 0:07:34 *
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1407/1407 [07:35<00:00,  3.09it/s]
Epoch [2/5]
 14%|█████████████████████▊                                                                                                                                         | 193/1407 [00:53<05:40,  3.57it/s]Iter: 1600, Train Loss: 0.38, Train Acc: 86.72%, Val Loss: 0.37, Val Acc: 88.88%, Time: 0:08:39 *
 28%|████████████████████████████████████████████▍                                                                                                                  | 393/1407 [01:59<04:43,  3.57it/s]Iter: 1800, Train Loss: 0.39, Train Acc: 89.06%, Val Loss: 0.36, Val Acc: 89.27%, Time: 0:09:44 *
 42%|███████████████████████████████████████████████████████████████████                                                                                            | 593/1407 [03:04<03:47,  3.57it/s]Iter: 2000, Train Loss: 0.45, Train Acc: 85.94%, Val Loss: 0.35, Val Acc: 89.70%, Time: 0:10:50 *
 56%|█████████████████████████████████████████████████████████████████████████████████████████▌                                                                     | 793/1407 [04:09<02:52,  3.56it/s]Iter: 2200, Train Loss: 0.35, Train Acc: 89.84%, Val Loss: 0.34, Val Acc: 89.96%, Time: 0:11:54 *
 71%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                              | 993/1407 [05:14<01:56,  3.57it/s]Iter: 2400, Train Loss: 0.36, Train Acc: 89.84%, Val Loss: 0.33, Val Acc: 90.08%, Time: 0:12:59 *
 85%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                        | 1193/1407 [06:19<00:59,  3.57it/s]Iter: 2600, Train Loss: 0.36, Train Acc: 89.84%, Val Loss: 0.33, Val Acc: 90.08%, Time: 0:14:04 *
 99%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍ | 1393/1407 [07:25<00:04,  3.30it/s]Iter: 2800, Train Loss: 0.37, Train Acc: 89.84%, Val Loss: 0.32, Val Acc: 90.35%, Time: 0:15:18 *
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1407/1407 [07:49<00:00,  3.00it/s]
Epoch [3/5]
 13%|█████████████████████                                                                                                                                          | 186/1407 [01:00<05:41,  3.58it/s]Iter: 3000, Train Loss: 0.34, Train Acc: 89.06%, Val Loss: 0.31, Val Acc: 90.49%, Time: 0:16:34 *
 27%|███████████████████████████████████████████▌                                                                                                                   | 386/1407 [02:04<04:45,  3.57it/s]Iter: 3200, Train Loss: 0.43, Train Acc: 89.06%, Val Loss: 0.31, Val Acc: 90.72%, Time: 0:17:39 *
 42%|██████████████████████████████████████████████████████████████████▏                                                                                            | 586/1407 [03:09<03:50,  3.57it/s]Iter: 3400, Train Loss: 0.40, Train Acc: 85.94%, Val Loss: 0.30, Val Acc: 90.80%, Time: 0:18:44 *
 56%|████████████████████████████████████████████████████████████████████████████████████████▊                                                                      | 786/1407 [04:14<02:53,  3.57it/s]Iter: 3600, Train Loss: 0.20, Train Acc: 94.53%, Val Loss: 0.30, Val Acc: 90.80%, Time: 0:19:49 *
 70%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                               | 986/1407 [05:19<01:58,  3.57it/s]Iter: 3800, Train Loss: 0.27, Train Acc: 89.06%, Val Loss: 0.30, Val Acc: 90.96%, Time: 0:20:54 *
 84%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                        | 1186/1407 [06:24<01:01,  3.57it/s]Iter: 4000, Train Loss: 0.22, Train Acc: 91.41%, Val Loss: 0.30, Val Acc: 90.92%, Time: 0:21:57 
 99%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋  | 1386/1407 [07:27<00:05,  3.57it/s]Iter: 4200, Train Loss: 0.33, Train Acc: 90.62%, Val Loss: 0.29, Val Acc: 91.19%, Time: 0:23:02 *
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1407/1407 [07:42<00:00,  3.04it/s]
Epoch [4/5]
 13%|████████████████████▏                                                                                                                                          | 179/1407 [00:49<05:43,  3.57it/s]Iter: 4400, Train Loss: 0.18, Train Acc: 92.97%, Val Loss: 0.29, Val Acc: 91.32%, Time: 0:24:06 *
 27%|██████████████████████████████████████████▊                                                                                                                    | 379/1407 [01:54<04:48,  3.57it/s]Iter: 4600, Train Loss: 0.26, Train Acc: 89.84%, Val Loss: 0.29, Val Acc: 91.30%, Time: 0:25:09 
 41%|█████████████████████████████████████████████████████████████████▍                                                                                             | 579/1407 [02:57<03:51,  3.58it/s]Iter: 4800, Train Loss: 0.24, Train Acc: 92.19%, Val Loss: 0.28, Val Acc: 91.36%, Time: 0:26:14 *
 55%|████████████████████████████████████████████████████████████████████████████████████████                                                                       | 779/1407 [04:02<02:55,  3.57it/s]Iter: 5000, Train Loss: 0.26, Train Acc: 93.75%, Val Loss: 0.28, Val Acc: 91.50%, Time: 0:27:17 
 70%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                                | 979/1407 [05:05<01:59,  3.57it/s]Iter: 5200, Train Loss: 0.31, Train Acc: 90.62%, Val Loss: 0.27, Val Acc: 91.50%, Time: 0:28:22 *
 84%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                         | 1179/1407 [06:10<01:03,  3.57it/s]Iter: 5400, Train Loss: 0.42, Train Acc: 85.94%, Val Loss: 0.28, Val Acc: 91.47%, Time: 0:29:25 
 98%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊   | 1379/1407 [07:13<00:07,  3.58it/s]Iter: 5600, Train Loss: 0.09, Train Acc: 99.22%, Val Loss: 0.27, Val Acc: 91.57%, Time: 0:30:28 
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1407/1407 [07:27<00:00,  3.14it/s]
Epoch [5/5]
 12%|███████████████████▍                                                                                                                                           | 172/1407 [00:47<05:45,  3.57it/s]Iter: 5800, Train Loss: 0.16, Train Acc: 94.53%, Val Loss: 0.28, Val Acc: 91.70%, Time: 0:31:30 
 26%|██████████████████████████████████████████                                                                                                                     | 372/1407 [01:50<04:49,  3.57it/s]Iter: 6000, Train Loss: 0.23, Train Acc: 92.19%, Val Loss: 0.28, Val Acc: 91.65%, Time: 0:32:33 
 41%|████████████████████████████████████████████████████████████████▋                                                                                              | 572/1407 [02:53<03:53,  3.57it/s]Iter: 6200, Train Loss: 0.18, Train Acc: 94.53%, Val Loss: 0.27, Val Acc: 91.94%, Time: 0:33:38 *
 55%|███████████████████████████████████████████████████████████████████████████████████████▏                                                                       | 772/1407 [03:58<02:57,  3.57it/s]Iter: 6400, Train Loss: 0.09, Train Acc: 98.44%, Val Loss: 0.27, Val Acc: 91.87%, Time: 0:34:43 *
 69%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                                 | 972/1407 [05:03<02:01,  3.57it/s]Iter: 6600, Train Loss: 0.25, Train Acc: 91.41%, Val Loss: 0.27, Val Acc: 91.92%, Time: 0:35:46 
 83%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                          | 1172/1407 [06:06<01:05,  3.57it/s]Iter: 6800, Train Loss: 0.21, Train Acc: 90.62%, Val Loss: 0.27, Val Acc: 91.85%, Time: 0:36:49 
 98%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████    | 1372/1407 [07:09<00:09,  3.57it/s]Iter: 7000, Train Loss: 0.25, Train Acc: 89.84%, Val Loss: 0.27, Val Acc: 91.91%, Time: 0:37:52 
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1407/1407 [07:26<00:00,  3.15it/s]
```

预测结果：

```css
Test Loss:  0.24, Test Acc: 92.35%
Precision, Recall and F1-Score...
               precision    recall  f1-score   support

      finance     0.9183    0.9110    0.9147      1000
       realty     0.9516    0.9440    0.9478      1000
       stocks     0.8756    0.8730    0.8743      1000
    education     0.9695    0.9550    0.9622      1000
      science     0.8769    0.8760    0.8764      1000
      society     0.9203    0.9240    0.9222      1000
     politics     0.8860    0.9330    0.9089      1000
       sports     0.9725    0.9560    0.9642      1000
         game     0.9388    0.9200    0.9293      1000
entertainment     0.9291    0.9430    0.9360      1000

     accuracy                         0.9235     10000
    macro avg     0.9239    0.9235    0.9236     10000
 weighted avg     0.9239    0.9235    0.9236     10000

Confusion Matrix...
[[911  15  42   0   8   8   9   2   4   1]
 [  6 944  16   1   3  10   8   2   3   7]
 [ 56  12 873   0  25   1  28   0   3   2]
 [  2   1   1 955   3  17  10   1   2   8]
 [  4   6  33   5 876  15  14   1  31  15]
 [  4   8   1  10   6 924  29   0   4  14]
 [  5   0  22   7  13  14 933   2   1   3]
 [  2   1   2   2   3   6  10 956   2  16]
 [  1   0   6   0  51   7   4   5 920   6]
 [  1   5   1   5  11   2   8  14  10 943]]
Time usage: 0:00:07
```

### 5.6 XLNet-Finetune

#### 5.6.1 XLNet配置

`chinese-xlnet-base`配置如下：

```json
{
  "architectures": [
    "XLNetLMHeadModel"
  ],
  "attn_type": "bi",
  "bi_data": false,
  "bos_token_id": 1,
  "clamp_len": -1,
  "d_head": 64,
  "d_inner": 3072,
  "d_model": 768,
  "dropout": 0.1,
  "end_n_top": 5,
  "eos_token_id": 2,
  "ff_activation": "relu",
  "initializer_range": 0.02,
  "layer_norm_eps": 1e-12,
  "mem_len": null,
  "model_type": "xlnet",
  "n_head": 12,
  "n_layer": 12,
  "output_past": true,
  "pad_token_id": 5,
  "reuse_len": null,
  "same_length": false,
  "start_n_top": 5,
  "summary_activation": "tanh",
  "summary_last_dropout": 0.1,
  "summary_type": "last",
  "summary_use_proj": true,
  "untie_r": true,
  "vocab_size": 32000
}
```

XLNet模型架构如下：

```css
Model(
  (xlnet): XLNetModel(
    (word_embedding): Embedding(32000, 768)
    (layer): ModuleList(
      (0-11): 12 x XLNetLayer(
        (rel_attn): XLNetRelativeAttention(
          (layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (ff): XLNetFeedForward(
          (layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (layer_1): Linear(in_features=768, out_features=3072, bias=True)
          (layer_2): Linear(in_features=3072, out_features=768, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation_function): ReLU()
        )
        (dropout): Dropout(p=0.1, inplace=False)
      )
    )
    (dropout): Dropout(p=0.1, inplace=False)
  )
  (fc): Linear(in_features=768, out_features=10, bias=True)
)
```

#### 5.6.2 训练和预测

直接执行如下命令进行训练:

```shell
python run_xlnet.py --model xlnet
```

训练过程和结果：

```css
Loading data for xlnet Model...
180000it [00:08, 20543.66it/s]
10000it [00:00, 22332.61it/s]
10000it [00:00, 22783.19it/s]
Epoch [1/5]
 14%|██████████████████████▌                                                                                                                                        | 200/1407 [00:57<05:46,  3.49it/s]Iter: 200, Train Loss: 0.50, Train Acc: 82.03%, Val Loss: 0.36, Val Acc: 88.60%, Time: 0:01:06 *
 28%|█████████████████████████████████████████████▏                                                                                                                 | 400/1407 [02:02<04:46,  3.51it/s]Iter: 400, Train Loss: 0.46, Train Acc: 87.50%, Val Loss: 0.31, Val Acc: 90.38%, Time: 0:02:12 *
 43%|███████████████████████████████████████████████████████████████████▊                                                                                           | 600/1407 [03:08<03:49,  3.52it/s]Iter: 600, Train Loss: 0.34, Train Acc: 87.50%, Val Loss: 0.28, Val Acc: 90.85%, Time: 0:03:17 *
 57%|██████████████████████████████████████████████████████████████████████████████████████████▍                                                                    | 800/1407 [04:13<02:52,  3.51it/s]Iter: 800, Train Loss: 0.20, Train Acc: 90.62%, Val Loss: 0.27, Val Acc: 91.40%, Time: 0:04:22 *
 71%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                             | 1000/1407 [05:18<01:55,  3.52it/s]Iter: 1000, Train Loss: 0.26, Train Acc: 91.41%, Val Loss: 0.27, Val Acc: 91.47%, Time: 0:05:27 
 85%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                       | 1200/1407 [06:23<00:58,  3.51it/s]Iter: 1200, Train Loss: 0.26, Train Acc: 90.62%, Val Loss: 0.25, Val Acc: 91.83%, Time: 0:06:32 *
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏| 1400/1407 [07:28<00:01,  3.51it/s]Iter: 1400, Train Loss: 0.31, Train Acc: 89.06%, Val Loss: 0.24, Val Acc: 92.38%, Time: 0:07:38 *
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1407/1407 [07:39<00:00,  3.06it/s]
Epoch [2/5]
 14%|█████████████████████▊                                                                                                                                         | 193/1407 [00:54<05:45,  3.51it/s]Iter: 1600, Train Loss: 0.22, Train Acc: 91.41%, Val Loss: 0.23, Val Acc: 92.47%, Time: 0:08:43 *
 28%|████████████████████████████████████████████▍                                                                                                                  | 393/1407 [02:00<04:48,  3.52it/s]Iter: 1800, Train Loss: 0.22, Train Acc: 94.53%, Val Loss: 0.23, Val Acc: 92.57%, Time: 0:09:49 *
 42%|███████████████████████████████████████████████████████████████████                                                                                            | 593/1407 [03:06<03:50,  3.53it/s]Iter: 2000, Train Loss: 0.19, Train Acc: 94.53%, Val Loss: 0.22, Val Acc: 92.80%, Time: 0:10:55 *
 56%|█████████████████████████████████████████████████████████████████████████████████████████▌                                                                     | 793/1407 [04:12<02:57,  3.45it/s]Iter: 2200, Train Loss: 0.25, Train Acc: 92.19%, Val Loss: 0.23, Val Acc: 92.68%, Time: 0:12:00 
 71%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                              | 993/1407 [05:17<01:59,  3.45it/s]Iter: 2400, Train Loss: 0.13, Train Acc: 95.31%, Val Loss: 0.22, Val Acc: 92.95%, Time: 0:13:05 
 85%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                        | 1193/1407 [06:22<01:00,  3.52it/s]Iter: 2600, Train Loss: 0.21, Train Acc: 92.19%, Val Loss: 0.21, Val Acc: 93.12%, Time: 0:14:10 *
 99%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍ | 1393/1407 [07:27<00:04,  3.48it/s]Iter: 2800, Train Loss: 0.18, Train Acc: 92.97%, Val Loss: 0.22, Val Acc: 93.04%, Time: 0:15:15 
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1407/1407 [07:39<00:00,  3.06it/s]
Epoch [3/5]
 13%|█████████████████████                                                                                                                                          | 186/1407 [00:52<05:48,  3.51it/s]Iter: 3000, Train Loss: 0.10, Train Acc: 97.66%, Val Loss: 0.22, Val Acc: 92.98%, Time: 0:16:19 
 27%|███████████████████████████████████████████▌                                                                                                                   | 386/1407 [01:57<04:51,  3.50it/s]Iter: 3200, Train Loss: 0.26, Train Acc: 92.97%, Val Loss: 0.22, Val Acc: 92.78%, Time: 0:17:24 
 42%|██████████████████████████████████████████████████████████████████▏                                                                                            | 586/1407 [03:01<03:53,  3.52it/s]Iter: 3400, Train Loss: 0.27, Train Acc: 92.97%, Val Loss: 0.21, Val Acc: 93.22%, Time: 0:18:29 *
 56%|████████████████████████████████████████████████████████████████████████████████████████▊                                                                      | 786/1407 [04:07<02:56,  3.51it/s]Iter: 3600, Train Loss: 0.13, Train Acc: 93.75%, Val Loss: 0.21, Val Acc: 93.29%, Time: 0:19:34 
 70%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                               | 986/1407 [05:11<01:59,  3.52it/s]Iter: 3800, Train Loss: 0.14, Train Acc: 93.75%, Val Loss: 0.22, Val Acc: 93.00%, Time: 0:20:38 
 84%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                        | 1186/1407 [06:16<01:02,  3.53it/s]Iter: 4000, Train Loss: 0.08, Train Acc: 98.44%, Val Loss: 0.22, Val Acc: 93.25%, Time: 0:21:43 
 99%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋  | 1386/1407 [07:20<00:05,  3.53it/s]Iter: 4200, Train Loss: 0.18, Train Acc: 92.19%, Val Loss: 0.21, Val Acc: 93.18%, Time: 0:22:48 *
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1407/1407 [07:35<00:00,  3.09it/s]
Epoch [4/5]
 13%|████████████████████▏                                                                                                                                          | 179/1407 [00:51<05:49,  3.51it/s]Iter: 4400, Train Loss: 0.15, Train Acc: 95.31%, Val Loss: 0.21, Val Acc: 93.36%, Time: 0:23:53 
 27%|██████████████████████████████████████████▊                                                                                                                    | 379/1407 [01:55<04:51,  3.52it/s]Iter: 4600, Train Loss: 0.13, Train Acc: 95.31%, Val Loss: 0.23, Val Acc: 93.17%, Time: 0:24:57 
 41%|█████████████████████████████████████████████████████████████████▍                                                                                             | 579/1407 [03:00<03:56,  3.51it/s]Iter: 4800, Train Loss: 0.06, Train Acc: 97.66%, Val Loss: 0.22, Val Acc: 93.26%, Time: 0:26:02 
 55%|████████████████████████████████████████████████████████████████████████████████████████                                                                       | 779/1407 [04:04<03:02,  3.44it/s]Iter: 5000, Train Loss: 0.08, Train Acc: 96.88%, Val Loss: 0.22, Val Acc: 93.56%, Time: 0:27:06 
 70%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                                | 979/1407 [05:09<02:02,  3.50it/s]Iter: 5200, Train Loss: 0.17, Train Acc: 92.19%, Val Loss: 0.23, Val Acc: 93.51%, Time: 0:28:11 
No optimization for a long time, auto-stopping...
 70%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                                | 979/1407 [05:17<02:18,  3.09it/s]
```

预测结果：

```css
Test Loss:  0.21, Test Acc: 94.24%
Precision, Recall and F1-Score...
               precision    recall  f1-score   support

      finance     0.9216    0.9520    0.9365      1000
       realty     0.9514    0.9590    0.9552      1000
       stocks     0.9338    0.8740    0.9029      1000
    education     0.9679    0.9660    0.9670      1000
      science     0.9066    0.9120    0.9093      1000
      society     0.9295    0.9490    0.9391      1000
     politics     0.9366    0.9160    0.9262      1000
       sports     0.9752    0.9820    0.9786      1000
         game     0.9692    0.9450    0.9570      1000
entertainment     0.9335    0.9690    0.9509      1000

     accuracy                         0.9424     10000
    macro avg     0.9425    0.9424    0.9423     10000
 weighted avg     0.9425    0.9424    0.9423     10000

Confusion Matrix...
[[952  10  15   3   4   7   6   1   0   2]
 [ 11 959   5   0   2   5   4   3   1  10]
 [ 53  19 874   1  30   2  16   1   1   3]
 [  2   1   1 966   1   8   8   1   1  11]
 [  5   3  16   4 912  14  11   2  20  13]
 [  1   7   3  11   4 949  17   0   0   8]
 [  7   8  18   7  13  23 916   1   0   7]
 [  1   1   1   0   1   4   0 982   1   9]
 [  0   0   2   3  34   6   0   4 945   6]
 [  1   0   1   3   5   3   0  12   6 969]]
Time usage: 0:00:07
```

### 5.7 Electra-Finetune

#### 5.7.1 Electra配置

`chinese-electra-base-discriminator`配置如下：

```json
{
  "attention_probs_dropout_prob": 0.1,
  "directionality": "bidi",
  "embedding_size": 768,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "electra",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "type_vocab_size": 2,
  "vocab_size": 21128
}
```

模型架构:

```css
Model(
  (electra): ElectraModel(
    (embeddings): ElectraEmbeddings(
      (word_embeddings): Embedding(21128, 768, padding_idx=0)
      (position_embeddings): Embedding(512, 768)
      (token_type_embeddings): Embedding(2, 768)
      (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (encoder): ElectraEncoder(
      (layer): ModuleList(
        (0-11): 12 x ElectraLayer(
          (attention): ElectraAttention(
            (self): ElectraSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (output): ElectraSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (intermediate): ElectraIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation()
          )
          (output): ElectraOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
    )
  )
  (fc): Linear(in_features=768, out_features=10, bias=True)
)
```

#### 5.7.2 训练和预测

直接调用如下命令：

```shell
python run_electra.py --model electra
```

训练过程和结果如下：

```css
Loading data for electra Model...
180000it [00:30, 5875.09it/s]
10000it [00:01, 5861.47it/s]
10000it [00:01, 5988.60it/s]
Epoch [1/5]
 14%|██████████████████████▌                                                                                                                                        | 200/1407 [00:45<04:33,  4.41it/s]Iter: 200, Train Loss: 0.61, Train Acc: 82.03%, Val Loss: 0.48, Val Acc: 86.78%, Time: 0:00:53 *
 28%|█████████████████████████████████████████████▏                                                                                                                 | 400/1407 [01:38<03:54,  4.30it/s]Iter: 400, Train Loss: 0.60, Train Acc: 82.81%, Val Loss: 0.38, Val Acc: 89.26%, Time: 0:01:46 *
 43%|███████████████████████████████████████████████████████████████████▊                                                                                           | 600/1407 [02:32<03:10,  4.23it/s]Iter: 600, Train Loss: 0.36, Train Acc: 89.84%, Val Loss: 0.33, Val Acc: 90.71%, Time: 0:02:39 *
 57%|██████████████████████████████████████████████████████████████████████████████████████████▍                                                                    | 800/1407 [03:25<02:21,  4.28it/s]Iter: 800, Train Loss: 0.23, Train Acc: 94.53%, Val Loss: 0.30, Val Acc: 91.28%, Time: 0:03:33 *
 71%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                             | 1000/1407 [04:19<01:35,  4.27it/s]Iter: 1000, Train Loss: 0.23, Train Acc: 93.75%, Val Loss: 0.29, Val Acc: 91.42%, Time: 0:04:26 *
 85%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                       | 1200/1407 [05:12<00:48,  4.27it/s]Iter: 1200, Train Loss: 0.25, Train Acc: 91.41%, Val Loss: 0.27, Val Acc: 91.70%, Time: 0:05:20 *
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏| 1400/1407 [06:06<00:01,  4.27it/s]Iter: 1400, Train Loss: 0.34, Train Acc: 91.41%, Val Loss: 0.26, Val Acc: 92.23%, Time: 0:06:14 *
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1407/1407 [06:15<00:00,  3.75it/s]
Epoch [2/5]
 14%|█████████████████████▊                                                                                                                                         | 193/1407 [00:45<04:44,  4.27it/s]Iter: 1600, Train Loss: 0.27, Train Acc: 90.62%, Val Loss: 0.27, Val Acc: 91.77%, Time: 0:07:07 
 28%|████████████████████████████████████████████▍                                                                                                                  | 393/1407 [01:37<03:57,  4.27it/s]Iter: 1800, Train Loss: 0.20, Train Acc: 93.75%, Val Loss: 0.25, Val Acc: 92.41%, Time: 0:08:00 *
 42%|███████████████████████████████████████████████████████████████████                                                                                            | 593/1407 [02:31<03:10,  4.27it/s]Iter: 2000, Train Loss: 0.30, Train Acc: 90.62%, Val Loss: 0.24, Val Acc: 92.40%, Time: 0:08:54 *
 56%|█████████████████████████████████████████████████████████████████████████████████████████▌                                                                     | 793/1407 [03:25<02:23,  4.28it/s]Iter: 2200, Train Loss: 0.15, Train Acc: 95.31%, Val Loss: 0.24, Val Acc: 92.71%, Time: 0:09:48 *
 71%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                              | 993/1407 [04:18<01:36,  4.27it/s]Iter: 2400, Train Loss: 0.18, Train Acc: 96.09%, Val Loss: 0.23, Val Acc: 92.76%, Time: 0:10:41 *
 85%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                        | 1193/1407 [05:12<00:50,  4.27it/s]Iter: 2600, Train Loss: 0.16, Train Acc: 96.09%, Val Loss: 0.23, Val Acc: 92.61%, Time: 0:11:35 *
 99%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍ | 1393/1407 [06:05<00:03,  4.27it/s]Iter: 2800, Train Loss: 0.26, Train Acc: 90.62%, Val Loss: 0.23, Val Acc: 92.71%, Time: 0:12:29 *
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1407/1407 [06:16<00:00,  3.74it/s]
Epoch [3/5]
 13%|█████████████████████                                                                                                                                          | 186/1407 [00:43<04:45,  4.28it/s]Iter: 3000, Train Loss: 0.14, Train Acc: 96.09%, Val Loss: 0.23, Val Acc: 92.98%, Time: 0:13:22 *
 27%|███████████████████████████████████████████▌                                                                                                                   | 386/1407 [01:37<03:58,  4.29it/s]Iter: 3200, Train Loss: 0.27, Train Acc: 92.97%, Val Loss: 0.23, Val Acc: 92.93%, Time: 0:14:15 
 42%|██████████████████████████████████████████████████████████████████▏                                                                                            | 586/1407 [02:29<03:11,  4.28it/s]Iter: 3400, Train Loss: 0.22, Train Acc: 92.97%, Val Loss: 0.22, Val Acc: 93.15%, Time: 0:15:09 *
 56%|████████████████████████████████████████████████████████████████████████████████████████▊                                                                      | 786/1407 [03:23<02:25,  4.27it/s]Iter: 3600, Train Loss: 0.09, Train Acc: 98.44%, Val Loss: 0.22, Val Acc: 93.08%, Time: 0:16:02 
 70%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                               | 986/1407 [04:16<01:38,  4.26it/s]Iter: 3800, Train Loss: 0.21, Train Acc: 92.19%, Val Loss: 0.22, Val Acc: 93.21%, Time: 0:16:55 *
 84%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                        | 1186/1407 [05:10<00:52,  4.24it/s]Iter: 4000, Train Loss: 0.09, Train Acc: 97.66%, Val Loss: 0.22, Val Acc: 92.79%, Time: 0:17:48 
 99%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋  | 1386/1407 [06:03<00:04,  4.26it/s]Iter: 4200, Train Loss: 0.21, Train Acc: 94.53%, Val Loss: 0.21, Val Acc: 93.29%, Time: 0:18:42 *
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1407/1407 [06:14<00:00,  3.75it/s]
Epoch [4/5]
 13%|████████████████████▏                                                                                                                                          | 179/1407 [00:41<04:47,  4.26it/s]Iter: 4400, Train Loss: 0.11, Train Acc: 98.44%, Val Loss: 0.21, Val Acc: 93.22%, Time: 0:19:34 
 27%|██████████████████████████████████████████▊                                                                                                                    | 379/1407 [01:34<04:00,  4.27it/s]Iter: 4600, Train Loss: 0.11, Train Acc: 96.09%, Val Loss: 0.22, Val Acc: 93.31%, Time: 0:20:27 
 41%|█████████████████████████████████████████████████████████████████▍                                                                                             | 579/1407 [02:27<03:13,  4.28it/s]Iter: 4800, Train Loss: 0.05, Train Acc: 98.44%, Val Loss: 0.22, Val Acc: 93.26%, Time: 0:21:20 
 55%|████████████████████████████████████████████████████████████████████████████████████████                                                                       | 779/1407 [03:20<02:26,  4.27it/s]Iter: 5000, Train Loss: 0.15, Train Acc: 95.31%, Val Loss: 0.22, Val Acc: 93.32%, Time: 0:22:13 
 70%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                                | 979/1407 [04:12<01:40,  4.27it/s]Iter: 5200, Train Loss: 0.23, Train Acc: 94.53%, Val Loss: 0.22, Val Acc: 93.27%, Time: 0:23:05 
No optimization for a long time, auto-stopping...
 70%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                                | 979/1407 [04:19<01:53,  3.78it/s]
```

预测结果如下：

```css
Test Loss:  0.22, Test Acc: 93.68%
Precision, Recall and F1-Score...
               precision    recall  f1-score   support

      finance     0.9008    0.9540    0.9267      1000
       realty     0.9671    0.9400    0.9533      1000
       stocks     0.9212    0.8770    0.8986      1000
    education     0.9574    0.9670    0.9622      1000
      science     0.9016    0.8980    0.8998      1000
      society     0.9068    0.9530    0.9293      1000
     politics     0.9167    0.9020    0.9093      1000
       sports     0.9828    0.9720    0.9774      1000
         game     0.9761    0.9380    0.9567      1000
entertainment     0.9425    0.9670    0.9546      1000

     accuracy                         0.9368     10000
    macro avg     0.9373    0.9368    0.9368     10000
 weighted avg     0.9373    0.9368    0.9368     10000

Confusion Matrix...
[[954   5  21   2   4   6   4   1   1   2]
 [ 20 940   9   1   2   9   5   4   1   9]
 [ 49  13 877   1  32   0  25   0   1   2]
 [  2   0   2 967   4  13   5   0   1   6]
 [ 11   2  14   5 898  21  18   2  15  14]
 [  3   8   2  11   3 953  11   0   1   8]
 [ 13   2  21  12  10  33 902   4   0   3]
 [  3   1   3   0   1   4   4 972   0  12]
 [  3   1   3   4  34   7   5   2 938   3]
 [  1   0   0   7   8   5   5   4   3 967]]
Time usage: 0:00:06
```

### 5.8 RoBERTa-Finetune

#### 5.8.1 RoBERTa配置

`chinese-roberta-wwm-ext`配置如下：

```json
{
  "architectures": [
    "BertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "bos_token_id": 0,
  "directionality": "bidi",
  "eos_token_id": 2,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "output_past": true,
  "pad_token_id": 0,
  "pooler_fc_size": 768,
  "pooler_num_attention_heads": 12,
  "pooler_num_fc_layers": 3,
  "pooler_size_per_head": 128,
  "pooler_type": "first_token_transform",
  "type_vocab_size": 2,
  "vocab_size": 21128
}

```

RoBERTa模型架构如下：

```css
Model(
  (roberta): RobertaModel(
    (embeddings): RobertaEmbeddings(
      (word_embeddings): Embedding(21128, 768, padding_idx=0)
      (position_embeddings): Embedding(512, 768, padding_idx=0)
      (token_type_embeddings): Embedding(2, 768)
      (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (encoder): RobertaEncoder(
      (layer): ModuleList(
        (0-11): 12 x RobertaLayer(
          (attention): RobertaAttention(
            (self): RobertaSdpaSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (output): RobertaSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (intermediate): RobertaIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation()
          )
          (output): RobertaOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
    )
    (pooler): RobertaPooler(
      (dense): Linear(in_features=768, out_features=768, bias=True)
      (activation): Tanh()
    )
  )
  (fc): Linear(in_features=768, out_features=10, bias=True)
)
```

#### 5.8.2 训练和预测

直接调用如下命令：

```shell
python run_roberta.py --model roberta
```

训练过程和结果如下：

```css
Loading data for roberta Model...
180000it [00:31, 5760.14it/s]
10000it [00:01, 6026.47it/s]
10000it [00:01, 5842.93it/s]
Some weights of RobertaModel were not initialized from the model checkpoint at /mnt/workspace/message_classification/BERT/pretrain_model/chinese-roberta-wwm-ext and are newly initialized: ['embeddings.LayerNorm.bias', 'embeddings.LayerNorm.weight', 'embeddings.position_embeddings.weight', 'embeddings.token_type_embeddings.weight', 'embeddings.word_embeddings.weight', 'encoder.layer.0.attention.output.LayerNorm.bias', 'encoder.layer.0.attention.output.LayerNorm.weight', 'encoder.layer.0.attention.output.dense.bias', 'encoder.layer.0.attention.output.dense.weight', 'encoder.layer.0.attention.self.key.bias', 'encoder.layer.0.attention.self.key.weight', 'encoder.layer.0.attention.self.query.bias', 'encoder.layer.0.attention.self.query.weight', 'encoder.layer.0.attention.self.value.bias', 'encoder.layer.0.attention.self.value.weight', 'encoder.layer.0.intermediate.dense.bias', 'encoder.layer.0.intermediate.dense.weight', 'encoder.layer.0.output.LayerNorm.bias', 'encoder.layer.0.output.LayerNorm.weight', 'encoder.layer.0.output.dense.bias', 'encoder.layer.0.output.dense.weight', 'encoder.layer.1.attention.output.LayerNorm.bias', 'encoder.layer.1.attention.output.LayerNorm.weight', 'encoder.layer.1.attention.output.dense.bias', 'encoder.layer.1.attention.output.dense.weight', 'encoder.layer.1.attention.self.key.bias', 'encoder.layer.1.attention.self.key.weight', 'encoder.layer.1.attention.self.query.bias', 'encoder.layer.1.attention.self.query.weight', 'encoder.layer.1.attention.self.value.bias', 'encoder.layer.1.attention.self.value.weight', 'encoder.layer.1.intermediate.dense.bias', 'encoder.layer.1.intermediate.dense.weight', 'encoder.layer.1.output.LayerNorm.bias', 'encoder.layer.1.output.LayerNorm.weight', 'encoder.layer.1.output.dense.bias', 'encoder.layer.1.output.dense.weight', 'encoder.layer.10.attention.output.LayerNorm.bias', 'encoder.layer.10.attention.output.LayerNorm.weight', 'encoder.layer.10.attention.output.dense.bias', 'encoder.layer.10.attention.output.dense.weight', 'encoder.layer.10.attention.self.key.bias', 'encoder.layer.10.attention.self.key.weight', 'encoder.layer.10.attention.self.query.bias', 'encoder.layer.10.attention.self.query.weight', 'encoder.layer.10.attention.self.value.bias', 'encoder.layer.10.attention.self.value.weight', 'encoder.layer.10.intermediate.dense.bias', 'encoder.layer.10.intermediate.dense.weight', 'encoder.layer.10.output.LayerNorm.bias', 'encoder.layer.10.output.LayerNorm.weight', 'encoder.layer.10.output.dense.bias', 'encoder.layer.10.output.dense.weight', 'encoder.layer.11.attention.output.LayerNorm.bias', 'encoder.layer.11.attention.output.LayerNorm.weight', 'encoder.layer.11.attention.output.dense.bias', 'encoder.layer.11.attention.output.dense.weight', 'encoder.layer.11.attention.self.key.bias', 'encoder.layer.11.attention.self.key.weight', 'encoder.layer.11.attention.self.query.bias', 'encoder.layer.11.attention.self.query.weight', 'encoder.layer.11.attention.self.value.bias', 'encoder.layer.11.attention.self.value.weight', 'encoder.layer.11.intermediate.dense.bias', 'encoder.layer.11.intermediate.dense.weight', 'encoder.layer.11.output.LayerNorm.bias', 'encoder.layer.11.output.LayerNorm.weight', 'encoder.layer.11.output.dense.bias', 'encoder.layer.11.output.dense.weight', 'encoder.layer.2.attention.output.LayerNorm.bias', 'encoder.layer.2.attention.output.LayerNorm.weight', 'encoder.layer.2.attention.output.dense.bias', 'encoder.layer.2.attention.output.dense.weight', 'encoder.layer.2.attention.self.key.bias', 'encoder.layer.2.attention.self.key.weight', 'encoder.layer.2.attention.self.query.bias', 'encoder.layer.2.attention.self.query.weight', 'encoder.layer.2.attention.self.value.bias', 'encoder.layer.2.attention.self.value.weight', 'encoder.layer.2.intermediate.dense.bias', 'encoder.layer.2.intermediate.dense.weight', 'encoder.layer.2.output.LayerNorm.bias', 'encoder.layer.2.output.LayerNorm.weight', 'encoder.layer.2.output.dense.bias', 'encoder.layer.2.output.dense.weight', 'encoder.layer.3.attention.output.LayerNorm.bias', 'encoder.layer.3.attention.output.LayerNorm.weight', 'encoder.layer.3.attention.output.dense.bias', 'encoder.layer.3.attention.output.dense.weight', 'encoder.layer.3.attention.self.key.bias', 'encoder.layer.3.attention.self.key.weight', 'encoder.layer.3.attention.self.query.bias', 'encoder.layer.3.attention.self.query.weight', 'encoder.layer.3.attention.self.value.bias', 'encoder.layer.3.attention.self.value.weight', 'encoder.layer.3.intermediate.dense.bias', 'encoder.layer.3.intermediate.dense.weight', 'encoder.layer.3.output.LayerNorm.bias', 'encoder.layer.3.output.LayerNorm.weight', 'encoder.layer.3.output.dense.bias', 'encoder.layer.3.output.dense.weight', 'encoder.layer.4.attention.output.LayerNorm.bias', 'encoder.layer.4.attention.output.LayerNorm.weight', 'encoder.layer.4.attention.output.dense.bias', 'encoder.layer.4.attention.output.dense.weight', 'encoder.layer.4.attention.self.key.bias', 'encoder.layer.4.attention.self.key.weight', 'encoder.layer.4.attention.self.query.bias', 'encoder.layer.4.attention.self.query.weight', 'encoder.layer.4.attention.self.value.bias', 'encoder.layer.4.attention.self.value.weight', 'encoder.layer.4.intermediate.dense.bias', 'encoder.layer.4.intermediate.dense.weight', 'encoder.layer.4.output.LayerNorm.bias', 'encoder.layer.4.output.LayerNorm.weight', 'encoder.layer.4.output.dense.bias', 'encoder.layer.4.output.dense.weight', 'encoder.layer.5.attention.output.LayerNorm.bias', 'encoder.layer.5.attention.output.LayerNorm.weight', 'encoder.layer.5.attention.output.dense.bias', 'encoder.layer.5.attention.output.dense.weight', 'encoder.layer.5.attention.self.key.bias', 'encoder.layer.5.attention.self.key.weight', 'encoder.layer.5.attention.self.query.bias', 'encoder.layer.5.attention.self.query.weight', 'encoder.layer.5.attention.self.value.bias', 'encoder.layer.5.attention.self.value.weight', 'encoder.layer.5.intermediate.dense.bias', 'encoder.layer.5.intermediate.dense.weight', 'encoder.layer.5.output.LayerNorm.bias', 'encoder.layer.5.output.LayerNorm.weight', 'encoder.layer.5.output.dense.bias', 'encoder.layer.5.output.dense.weight', 'encoder.layer.6.attention.output.LayerNorm.bias', 'encoder.layer.6.attention.output.LayerNorm.weight', 'encoder.layer.6.attention.output.dense.bias', 'encoder.layer.6.attention.output.dense.weight', 'encoder.layer.6.attention.self.key.bias', 'encoder.layer.6.attention.self.key.weight', 'encoder.layer.6.attention.self.query.bias', 'encoder.layer.6.attention.self.query.weight', 'encoder.layer.6.attention.self.value.bias', 'encoder.layer.6.attention.self.value.weight', 'encoder.layer.6.intermediate.dense.bias', 'encoder.layer.6.intermediate.dense.weight', 'encoder.layer.6.output.LayerNorm.bias', 'encoder.layer.6.output.LayerNorm.weight', 'encoder.layer.6.output.dense.bias', 'encoder.layer.6.output.dense.weight', 'encoder.layer.7.attention.output.LayerNorm.bias', 'encoder.layer.7.attention.output.LayerNorm.weight', 'encoder.layer.7.attention.output.dense.bias', 'encoder.layer.7.attention.output.dense.weight', 'encoder.layer.7.attention.self.key.bias', 'encoder.layer.7.attention.self.key.weight', 'encoder.layer.7.attention.self.query.bias', 'encoder.layer.7.attention.self.query.weight', 'encoder.layer.7.attention.self.value.bias', 'encoder.layer.7.attention.self.value.weight', 'encoder.layer.7.intermediate.dense.bias', 'encoder.layer.7.intermediate.dense.weight', 'encoder.layer.7.output.LayerNorm.bias', 'encoder.layer.7.output.LayerNorm.weight', 'encoder.layer.7.output.dense.bias', 'encoder.layer.7.output.dense.weight', 'encoder.layer.8.attention.output.LayerNorm.bias', 'encoder.layer.8.attention.output.LayerNorm.weight', 'encoder.layer.8.attention.output.dense.bias', 'encoder.layer.8.attention.output.dense.weight', 'encoder.layer.8.attention.self.key.bias', 'encoder.layer.8.attention.self.key.weight', 'encoder.layer.8.attention.self.query.bias', 'encoder.layer.8.attention.self.query.weight', 'encoder.layer.8.attention.self.value.bias', 'encoder.layer.8.attention.self.value.weight', 'encoder.layer.8.intermediate.dense.bias', 'encoder.layer.8.intermediate.dense.weight', 'encoder.layer.8.output.LayerNorm.bias', 'encoder.layer.8.output.LayerNorm.weight', 'encoder.layer.8.output.dense.bias', 'encoder.layer.8.output.dense.weight', 'encoder.layer.9.attention.output.LayerNorm.bias', 'encoder.layer.9.attention.output.LayerNorm.weight', 'encoder.layer.9.attention.output.dense.bias', 'encoder.layer.9.attention.output.dense.weight', 'encoder.layer.9.attention.self.key.bias', 'encoder.layer.9.attention.self.key.weight', 'encoder.layer.9.attention.self.query.bias', 'encoder.layer.9.attention.self.query.weight', 'encoder.layer.9.attention.self.value.bias', 'encoder.layer.9.attention.self.value.weight', 'encoder.layer.9.intermediate.dense.bias', 'encoder.layer.9.intermediate.dense.weight', 'encoder.layer.9.output.LayerNorm.bias', 'encoder.layer.9.output.LayerNorm.weight', 'encoder.layer.9.output.dense.bias', 'encoder.layer.9.output.dense.weight', 'pooler.dense.bias', 'pooler.dense.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Epoch [1/5]
 14%|██████████████████████▌                                                                                                                                        | 200/1407 [00:46<04:39,  4.33it/s]Iter: 200, Train Loss: 0.96, Train Acc: 64.84%, Val Loss: 0.77, Val Acc: 74.88%, Time: 0:00:54 *
 28%|█████████████████████████████████████████████▏                                                                                                                 | 400/1407 [01:40<03:54,  4.30it/s]Iter: 400, Train Loss: 0.72, Train Acc: 76.56%, Val Loss: 0.57, Val Acc: 81.54%, Time: 0:01:48 *
 43%|███████████████████████████████████████████████████████████████████▊                                                                                           | 600/1407 [02:33<03:06,  4.33it/s]Iter: 600, Train Loss: 0.56, Train Acc: 84.38%, Val Loss: 0.54, Val Acc: 82.32%, Time: 0:02:40 *
 57%|██████████████████████████████████████████████████████████████████████████████████████████▍                                                                    | 800/1407 [03:26<02:20,  4.33it/s]Iter: 800, Train Loss: 0.42, Train Acc: 85.16%, Val Loss: 0.48, Val Acc: 85.09%, Time: 0:03:34 *
 71%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                             | 1000/1407 [04:19<01:33,  4.36it/s]Iter: 1000, Train Loss: 0.32, Train Acc: 90.62%, Val Loss: 0.47, Val Acc: 85.36%, Time: 0:04:27 *
 85%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                       | 1200/1407 [05:12<00:47,  4.34it/s]Iter: 1200, Train Loss: 0.45, Train Acc: 82.81%, Val Loss: 0.46, Val Acc: 85.38%, Time: 0:05:20 *
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏| 1400/1407 [06:05<00:01,  4.31it/s]Iter: 1400, Train Loss: 0.55, Train Acc: 81.25%, Val Loss: 0.42, Val Acc: 86.73%, Time: 0:06:13 *
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1407/1407 [06:14<00:00,  3.76it/s]
Epoch [2/5]
 14%|█████████████████████▊                                                                                                                                         | 193/1407 [00:44<04:39,  4.34it/s]Iter: 1600, Train Loss: 0.37, Train Acc: 87.50%, Val Loss: 0.44, Val Acc: 86.36%, Time: 0:07:05 
 28%|████████████████████████████████████████████▍                                                                                                                  | 393/1407 [01:36<03:52,  4.35it/s]Iter: 1800, Train Loss: 0.28, Train Acc: 94.53%, Val Loss: 0.39, Val Acc: 87.69%, Time: 0:07:58 *
 42%|███████████████████████████████████████████████████████████████████                                                                                            | 593/1407 [02:29<03:08,  4.32it/s]Iter: 2000, Train Loss: 0.40, Train Acc: 85.94%, Val Loss: 0.38, Val Acc: 87.88%, Time: 0:08:52 *
 56%|█████████████████████████████████████████████████████████████████████████████████████████▌                                                                     | 793/1407 [03:23<02:21,  4.33it/s]Iter: 2200, Train Loss: 0.22, Train Acc: 94.53%, Val Loss: 0.37, Val Acc: 88.53%, Time: 0:09:45 *
 71%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                              | 993/1407 [04:16<01:35,  4.35it/s]Iter: 2400, Train Loss: 0.25, Train Acc: 92.19%, Val Loss: 0.39, Val Acc: 87.96%, Time: 0:10:37 
 85%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                        | 1193/1407 [05:08<00:49,  4.33it/s]Iter: 2600, Train Loss: 0.34, Train Acc: 91.41%, Val Loss: 0.37, Val Acc: 88.49%, Time: 0:11:30 
 99%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍ | 1393/1407 [06:00<00:03,  4.33it/s]Iter: 2800, Train Loss: 0.51, Train Acc: 83.59%, Val Loss: 0.37, Val Acc: 88.64%, Time: 0:12:22 
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1407/1407 [06:10<00:00,  3.80it/s]
Epoch [3/5]
 13%|█████████████████████                                                                                                                                          | 186/1407 [00:42<04:41,  4.34it/s]Iter: 3000, Train Loss: 0.29, Train Acc: 91.41%, Val Loss: 0.35, Val Acc: 89.17%, Time: 0:13:15 *
 27%|███████████████████████████████████████████▌                                                                                                                   | 386/1407 [01:35<03:55,  4.34it/s]Iter: 3200, Train Loss: 0.48, Train Acc: 87.50%, Val Loss: 0.37, Val Acc: 88.46%, Time: 0:14:07 
 42%|██████████████████████████████████████████████████████████████████▏                                                                                            | 586/1407 [02:28<03:09,  4.33it/s]Iter: 3400, Train Loss: 0.32, Train Acc: 88.28%, Val Loss: 0.34, Val Acc: 89.32%, Time: 0:15:01 *
 56%|████████████████████████████████████████████████████████████████████████████████████████▊                                                                      | 786/1407 [03:21<02:23,  4.32it/s]Iter: 3600, Train Loss: 0.21, Train Acc: 94.53%, Val Loss: 0.35, Val Acc: 89.07%, Time: 0:15:53 
 70%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                               | 986/1407 [04:14<01:37,  4.34it/s]Iter: 3800, Train Loss: 0.31, Train Acc: 89.84%, Val Loss: 0.37, Val Acc: 88.36%, Time: 0:16:46 
 84%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                        | 1186/1407 [05:06<00:51,  4.33it/s]Iter: 4000, Train Loss: 0.25, Train Acc: 92.19%, Val Loss: 0.35, Val Acc: 89.11%, Time: 0:17:38 
 99%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋  | 1386/1407 [05:58<00:04,  4.35it/s]Iter: 4200, Train Loss: 0.39, Train Acc: 88.28%, Val Loss: 0.37, Val Acc: 88.55%, Time: 0:18:30 
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1407/1407 [06:09<00:00,  3.80it/s]
Epoch [4/5]
 13%|████████████████████▏                                                                                                                                          | 179/1407 [00:41<04:42,  4.34it/s]Iter: 4400, Train Loss: 0.18, Train Acc: 92.97%, Val Loss: 0.33, Val Acc: 89.28%, Time: 0:19:24 *
 27%|██████████████████████████████████████████▊                                                                                                                    | 379/1407 [01:34<03:57,  4.32it/s]Iter: 4600, Train Loss: 0.28, Train Acc: 89.06%, Val Loss: 0.35, Val Acc: 88.85%, Time: 0:20:16 
 41%|█████████████████████████████████████████████████████████████████▍                                                                                             | 579/1407 [02:27<03:09,  4.36it/s]Iter: 4800, Train Loss: 0.16, Train Acc: 95.31%, Val Loss: 0.34, Val Acc: 89.38%, Time: 0:21:09 
 55%|████████████████████████████████████████████████████████████████████████████████████████                                                                       | 779/1407 [03:19<02:24,  4.36it/s]Iter: 5000, Train Loss: 0.32, Train Acc: 89.06%, Val Loss: 0.37, Val Acc: 88.50%, Time: 0:22:01 
 70%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                                | 979/1407 [04:11<01:39,  4.32it/s]Iter: 5200, Train Loss: 0.35, Train Acc: 88.28%, Val Loss: 0.35, Val Acc: 88.81%, Time: 0:22:53 
 84%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                         | 1179/1407 [05:04<00:52,  4.32it/s]Iter: 5400, Train Loss: 0.38, Train Acc: 86.72%, Val Loss: 0.34, Val Acc: 89.59%, Time: 0:23:46 
No optimization for a long time, auto-stopping...
 84%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                         | 1179/1407 [05:10<01:00,  3.79it/s]
```

预测结果如下：

```css
Test Loss:  0.33, Test Acc: 89.69%
Precision, Recall and F1-Score...
               precision    recall  f1-score   support

      finance     0.8655    0.8880    0.8766      1000
       realty     0.9200    0.9080    0.9139      1000
       stocks     0.8290    0.8240    0.8265      1000
    education     0.9284    0.9470    0.9376      1000
      science     0.8670    0.8080    0.8364      1000
      society     0.8480    0.9370    0.8903      1000
     politics     0.9197    0.8250    0.8698      1000
       sports     0.9605    0.9730    0.9667      1000
         game     0.9322    0.9210    0.9266      1000
entertainment     0.9037    0.9380    0.9205      1000

     accuracy                         0.8969     10000
    macro avg     0.8974    0.8969    0.8965     10000
 weighted avg     0.8974    0.8969    0.8965     10000

Confusion Matrix...
[[888  13  47   6  13  14   9   2   4   4]
 [ 21 908  25   2   4  21   2   6   1  10]
 [ 83  24 824   2  38   4  19   1   2   3]
 [  2   3   2 947   4  21   5   2   0  14]
 [  6   8  48  13 808  28  17   5  41  26]
 [  3  11   2  19   1 937   9   0   3  15]
 [ 14  10  38  12  23  52 825   4  10  12]
 [  1   1   3   1   2   5   4 973   1   9]
 [  3   2   3  13  32  14   3   2 921   7]
 [  5   7   2   5   7   9   4  18   5 938]]
Time usage: 0:00:06
```

### 5.9 MacBERT-Finetune

#### 5.9.1 MacBERT配置

`chinese-macbert-base`配置如下：

```json
{
  "architectures": [
    "BertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "directionality": "bidi",
  "gradient_checkpointing": false,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "pooler_fc_size": 768,
  "pooler_num_attention_heads": 12,
  "pooler_num_fc_layers": 3,
  "pooler_size_per_head": 128,
  "pooler_type": "first_token_transform",
  "type_vocab_size": 2,
  "vocab_size": 21128
}
```

MacBert模型架构如下：

```css
Model(
  (macbert): BertModel(
    (embeddings): BertEmbeddings(
      (word_embeddings): Embedding(21128, 768, padding_idx=0)
      (position_embeddings): Embedding(512, 768)
      (token_type_embeddings): Embedding(2, 768)
      (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (encoder): BertEncoder(
      (layer): ModuleList(
        (0-11): 12 x BertLayer(
          (attention): BertAttention(
            (self): BertSdpaSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation()
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
    )
    (pooler): BertPooler(
      (dense): Linear(in_features=768, out_features=768, bias=True)
      (activation): Tanh()
    )
  )
  (fc): Linear(in_features=768, out_features=10, bias=True)
)
```

#### 5.9.1 训练和预测

直接调用如下命令：

```shell
python run_macbert.py --model macbert
```

训练过程和结果如下：

```css
Loading data for macbert Model...
180000it [00:30, 5837.49it/s]
10000it [00:01, 6244.91it/s]
10000it [00:01, 6254.14it/s]
Epoch [1/5]
 14%|██████████████████████▌                                                                                                                                        | 200/1407 [00:43<04:21,  4.62it/s]Iter: 200, Train Loss: 0.27, Train Acc: 91.41%, Val Loss: 0.28, Val Acc: 91.31%, Time: 0:00:50 *
 28%|█████████████████████████████████████████████▏                                                                                                                 | 400/1407 [01:33<03:42,  4.53it/s]Iter: 400, Train Loss: 0.36, Train Acc: 89.06%, Val Loss: 0.24, Val Acc: 92.47%, Time: 0:01:41 *
 43%|███████████████████████████████████████████████████████████████████▊                                                                                           | 600/1407 [02:24<03:00,  4.47it/s]Iter: 600, Train Loss: 0.26, Train Acc: 92.19%, Val Loss: 0.23, Val Acc: 92.41%, Time: 0:02:34 *
 57%|██████████████████████████████████████████████████████████████████████████████████████████▍                                                                    | 800/1407 [03:17<02:15,  4.49it/s]Iter: 800, Train Loss: 0.16, Train Acc: 93.75%, Val Loss: 0.21, Val Acc: 93.09%, Time: 0:03:25 *
 71%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                             | 1000/1407 [04:08<01:30,  4.48it/s]Iter: 1000, Train Loss: 0.18, Train Acc: 92.97%, Val Loss: 0.21, Val Acc: 93.06%, Time: 0:04:16 *
 85%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                       | 1200/1407 [05:00<00:46,  4.49it/s]Iter: 1200, Train Loss: 0.20, Train Acc: 92.97%, Val Loss: 0.20, Val Acc: 93.16%, Time: 0:05:07 *
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏| 1400/1407 [05:51<00:01,  4.48it/s]Iter: 1400, Train Loss: 0.26, Train Acc: 93.75%, Val Loss: 0.19, Val Acc: 93.74%, Time: 0:05:59 *
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1407/1407 [05:59<00:00,  3.91it/s]
Epoch [2/5]
 14%|█████████████████████▊                                                                                                                                         | 193/1407 [00:42<04:32,  4.46it/s]Iter: 1600, Train Loss: 0.23, Train Acc: 92.97%, Val Loss: 0.19, Val Acc: 93.79%, Time: 0:06:50 *
 28%|████████████████████████████████████████████▍                                                                                                                  | 393/1407 [01:33<03:46,  4.49it/s]Iter: 1800, Train Loss: 0.13, Train Acc: 95.31%, Val Loss: 0.19, Val Acc: 93.74%, Time: 0:07:41 *
 42%|███████████████████████████████████████████████████████████████████                                                                                            | 593/1407 [02:25<03:02,  4.47it/s]Iter: 2000, Train Loss: 0.14, Train Acc: 95.31%, Val Loss: 0.18, Val Acc: 94.12%, Time: 0:08:32 *
 56%|█████████████████████████████████████████████████████████████████████████████████████████▌                                                                     | 793/1407 [03:16<02:16,  4.49it/s]Iter: 2200, Train Loss: 0.11, Train Acc: 96.09%, Val Loss: 0.20, Val Acc: 93.94%, Time: 0:09:22 
 71%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                              | 993/1407 [04:06<01:32,  4.49it/s]Iter: 2400, Train Loss: 0.08, Train Acc: 98.44%, Val Loss: 0.19, Val Acc: 94.04%, Time: 0:10:12 
 85%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                        | 1193/1407 [04:56<00:47,  4.48it/s]Iter: 2600, Train Loss: 0.13, Train Acc: 94.53%, Val Loss: 0.19, Val Acc: 94.01%, Time: 0:11:02 
 99%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍ | 1393/1407 [05:46<00:03,  4.50it/s]Iter: 2800, Train Loss: 0.09, Train Acc: 97.66%, Val Loss: 0.18, Val Acc: 94.18%, Time: 0:11:54 *
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1407/1407 [05:56<00:00,  3.94it/s]
Epoch [3/5]
 13%|█████████████████████                                                                                                                                          | 186/1407 [00:41<04:31,  4.50it/s]Iter: 3000, Train Loss: 0.08, Train Acc: 97.66%, Val Loss: 0.19, Val Acc: 94.02%, Time: 0:12:44 
 27%|███████████████████████████████████████████▌                                                                                                                   | 386/1407 [01:31<03:47,  4.48it/s]Iter: 3200, Train Loss: 0.15, Train Acc: 96.09%, Val Loss: 0.20, Val Acc: 94.07%, Time: 0:13:34 
 42%|██████████████████████████████████████████████████████████████████▏                                                                                            | 586/1407 [02:21<03:02,  4.50it/s]Iter: 3400, Train Loss: 0.19, Train Acc: 96.09%, Val Loss: 0.20, Val Acc: 93.99%, Time: 0:14:24 
 56%|████████████████████████████████████████████████████████████████████████████████████████▊                                                                      | 786/1407 [03:11<02:18,  4.50it/s]Iter: 3600, Train Loss: 0.04, Train Acc: 99.22%, Val Loss: 0.20, Val Acc: 94.17%, Time: 0:15:14 
 70%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                               | 986/1407 [04:01<01:33,  4.50it/s]Iter: 3800, Train Loss: 0.14, Train Acc: 93.75%, Val Loss: 0.21, Val Acc: 94.18%, Time: 0:16:04 
No optimization for a long time, auto-stopping...
 70%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                               | 986/1407 [04:07<01:45,  3.98it/s]
```

预测结果：

```css
Test Loss:  0.19, Test Acc: 94.74%
Precision, Recall and F1-Score...
               precision    recall  f1-score   support

      finance     0.9354    0.9410    0.9382      1000
       realty     0.9479    0.9650    0.9564      1000
       stocks     0.9468    0.8900    0.9175      1000
    education     0.9765    0.9560    0.9661      1000
      science     0.9156    0.9220    0.9188      1000
      society     0.9220    0.9580    0.9397      1000
     politics     0.9370    0.9220    0.9294      1000
       sports     0.9860    0.9840    0.9850      1000
         game     0.9784    0.9520    0.9650      1000
entertainment     0.9318    0.9840    0.9572      1000

     accuracy                         0.9474     10000
    macro avg     0.9477    0.9474    0.9473     10000
 weighted avg     0.9477    0.9474    0.9473     10000

Confusion Matrix...
[[941  14  20   1   2   9   9   2   0   2]
 [  8 965   4   1   3   5   3   3   0   8]
 [ 40  17 890   0  29   2  17   1   0   4]
 [  1   2   1 956   1  17   9   0   3  10]
 [  6   3   9   2 922  15  14   1  13  15]
 [  0   8   1   8   3 958   8   0   1  13]
 [  9   5  13   6  11  25 922   2   1   6]
 [  1   3   1   0   1   2   1 984   0   7]
 [  0   1   1   2  29   5   1   2 952   7]
 [  0   0   0   3   6   1   0   3   3 984]]
Time usage: 0:00:06
```

### 5.10 ERNIE3.0-Finetune

#### 5.10.1 ERNIE3.0配置

`ernie-3.0-base-zh`配置如下：

```json
{
    "attention_probs_dropout_prob": 0.1,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "hidden_size": 768,
    "initializer_range": 0.02,
    "max_position_embeddings": 2048,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "task_type_vocab_size": 3,
    "type_vocab_size": 4,
    "use_task_id": true,
    "vocab_size": 40000,
    "layer_norm_eps": 1e-05,
    "model_type": "ernie",
    "architectures": [
        "ErnieForMaskedLM"
    ],
    "intermediate_size": 3072
}
```

ERNIE3.0模型架构如下：

```css
Model(
  (ernie): BertModel(
    (embeddings): BertEmbeddings(
      (word_embeddings): Embedding(40000, 768, padding_idx=0)
      (position_embeddings): Embedding(2048, 768)
      (token_type_embeddings): Embedding(4, 768)
      (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (encoder): BertEncoder(
      (layer): ModuleList(
        (0-11): 12 x BertLayer(
          (attention): BertAttention(
            (self): BertSdpaSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation()
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
    )
    (pooler): BertPooler(
      (dense): Linear(in_features=768, out_features=768, bias=True)
      (activation): Tanh()
    )
  )
  (fc): Linear(in_features=768, out_features=10, bias=True)
)
```

直接调用如下命令：

```shell
python run_ernie.py --model ernie
```

训练过程和结果如下：

```css
Loading data for ernie Model...
180000it [00:29, 6021.07it/s]
10000it [00:01, 5882.80it/s]
10000it [00:01, 5823.33it/s]
Some weights of BertModel were not initialized from the model checkpoint at /mnt/workspace/message_classification/BERT/pretrain_model/ernie-3.0-base-zh and are newly initialized: ['embeddings.LayerNorm.bias', 'embeddings.LayerNorm.weight', 'embeddings.position_embeddings.weight', 'embeddings.token_type_embeddings.weight', 'embeddings.word_embeddings.weight', 'encoder.layer.0.attention.output.LayerNorm.bias', 'encoder.layer.0.attention.output.LayerNorm.weight', 'encoder.layer.0.attention.output.dense.bias', 'encoder.layer.0.attention.output.dense.weight', 'encoder.layer.0.attention.self.key.bias', 'encoder.layer.0.attention.self.key.weight', 'encoder.layer.0.attention.self.query.bias', 'encoder.layer.0.attention.self.query.weight', 'encoder.layer.0.attention.self.value.bias', 'encoder.layer.0.attention.self.value.weight', 'encoder.layer.0.intermediate.dense.bias', 'encoder.layer.0.intermediate.dense.weight', 'encoder.layer.0.output.LayerNorm.bias', 'encoder.layer.0.output.LayerNorm.weight', 'encoder.layer.0.output.dense.bias', 'encoder.layer.0.output.dense.weight', 'encoder.layer.1.attention.output.LayerNorm.bias', 'encoder.layer.1.attention.output.LayerNorm.weight', 'encoder.layer.1.attention.output.dense.bias', 'encoder.layer.1.attention.output.dense.weight', 'encoder.layer.1.attention.self.key.bias', 'encoder.layer.1.attention.self.key.weight', 'encoder.layer.1.attention.self.query.bias', 'encoder.layer.1.attention.self.query.weight', 'encoder.layer.1.attention.self.value.bias', 'encoder.layer.1.attention.self.value.weight', 'encoder.layer.1.intermediate.dense.bias', 'encoder.layer.1.intermediate.dense.weight', 'encoder.layer.1.output.LayerNorm.bias', 'encoder.layer.1.output.LayerNorm.weight', 'encoder.layer.1.output.dense.bias', 'encoder.layer.1.output.dense.weight', 'encoder.layer.10.attention.output.LayerNorm.bias', 'encoder.layer.10.attention.output.LayerNorm.weight', 'encoder.layer.10.attention.output.dense.bias', 'encoder.layer.10.attention.output.dense.weight', 'encoder.layer.10.attention.self.key.bias', 'encoder.layer.10.attention.self.key.weight', 'encoder.layer.10.attention.self.query.bias', 'encoder.layer.10.attention.self.query.weight', 'encoder.layer.10.attention.self.value.bias', 'encoder.layer.10.attention.self.value.weight', 'encoder.layer.10.intermediate.dense.bias', 'encoder.layer.10.intermediate.dense.weight', 'encoder.layer.10.output.LayerNorm.bias', 'encoder.layer.10.output.LayerNorm.weight', 'encoder.layer.10.output.dense.bias', 'encoder.layer.10.output.dense.weight', 'encoder.layer.11.attention.output.LayerNorm.bias', 'encoder.layer.11.attention.output.LayerNorm.weight', 'encoder.layer.11.attention.output.dense.bias', 'encoder.layer.11.attention.output.dense.weight', 'encoder.layer.11.attention.self.key.bias', 'encoder.layer.11.attention.self.key.weight', 'encoder.layer.11.attention.self.query.bias', 'encoder.layer.11.attention.self.query.weight', 'encoder.layer.11.attention.self.value.bias', 'encoder.layer.11.attention.self.value.weight', 'encoder.layer.11.intermediate.dense.bias', 'encoder.layer.11.intermediate.dense.weight', 'encoder.layer.11.output.LayerNorm.bias', 'encoder.layer.11.output.LayerNorm.weight', 'encoder.layer.11.output.dense.bias', 'encoder.layer.11.output.dense.weight', 'encoder.layer.2.attention.output.LayerNorm.bias', 'encoder.layer.2.attention.output.LayerNorm.weight', 'encoder.layer.2.attention.output.dense.bias', 'encoder.layer.2.attention.output.dense.weight', 'encoder.layer.2.attention.self.key.bias', 'encoder.layer.2.attention.self.key.weight', 'encoder.layer.2.attention.self.query.bias', 'encoder.layer.2.attention.self.query.weight', 'encoder.layer.2.attention.self.value.bias', 'encoder.layer.2.attention.self.value.weight', 'encoder.layer.2.intermediate.dense.bias', 'encoder.layer.2.intermediate.dense.weight', 'encoder.layer.2.output.LayerNorm.bias', 'encoder.layer.2.output.LayerNorm.weight', 'encoder.layer.2.output.dense.bias', 'encoder.layer.2.output.dense.weight', 'encoder.layer.3.attention.output.LayerNorm.bias', 'encoder.layer.3.attention.output.LayerNorm.weight', 'encoder.layer.3.attention.output.dense.bias', 'encoder.layer.3.attention.output.dense.weight', 'encoder.layer.3.attention.self.key.bias', 'encoder.layer.3.attention.self.key.weight', 'encoder.layer.3.attention.self.query.bias', 'encoder.layer.3.attention.self.query.weight', 'encoder.layer.3.attention.self.value.bias', 'encoder.layer.3.attention.self.value.weight', 'encoder.layer.3.intermediate.dense.bias', 'encoder.layer.3.intermediate.dense.weight', 'encoder.layer.3.output.LayerNorm.bias', 'encoder.layer.3.output.LayerNorm.weight', 'encoder.layer.3.output.dense.bias', 'encoder.layer.3.output.dense.weight', 'encoder.layer.4.attention.output.LayerNorm.bias', 'encoder.layer.4.attention.output.LayerNorm.weight', 'encoder.layer.4.attention.output.dense.bias', 'encoder.layer.4.attention.output.dense.weight', 'encoder.layer.4.attention.self.key.bias', 'encoder.layer.4.attention.self.key.weight', 'encoder.layer.4.attention.self.query.bias', 'encoder.layer.4.attention.self.query.weight', 'encoder.layer.4.attention.self.value.bias', 'encoder.layer.4.attention.self.value.weight', 'encoder.layer.4.intermediate.dense.bias', 'encoder.layer.4.intermediate.dense.weight', 'encoder.layer.4.output.LayerNorm.bias', 'encoder.layer.4.output.LayerNorm.weight', 'encoder.layer.4.output.dense.bias', 'encoder.layer.4.output.dense.weight', 'encoder.layer.5.attention.output.LayerNorm.bias', 'encoder.layer.5.attention.output.LayerNorm.weight', 'encoder.layer.5.attention.output.dense.bias', 'encoder.layer.5.attention.output.dense.weight', 'encoder.layer.5.attention.self.key.bias', 'encoder.layer.5.attention.self.key.weight', 'encoder.layer.5.attention.self.query.bias', 'encoder.layer.5.attention.self.query.weight', 'encoder.layer.5.attention.self.value.bias', 'encoder.layer.5.attention.self.value.weight', 'encoder.layer.5.intermediate.dense.bias', 'encoder.layer.5.intermediate.dense.weight', 'encoder.layer.5.output.LayerNorm.bias', 'encoder.layer.5.output.LayerNorm.weight', 'encoder.layer.5.output.dense.bias', 'encoder.layer.5.output.dense.weight', 'encoder.layer.6.attention.output.LayerNorm.bias', 'encoder.layer.6.attention.output.LayerNorm.weight', 'encoder.layer.6.attention.output.dense.bias', 'encoder.layer.6.attention.output.dense.weight', 'encoder.layer.6.attention.self.key.bias', 'encoder.layer.6.attention.self.key.weight', 'encoder.layer.6.attention.self.query.bias', 'encoder.layer.6.attention.self.query.weight', 'encoder.layer.6.attention.self.value.bias', 'encoder.layer.6.attention.self.value.weight', 'encoder.layer.6.intermediate.dense.bias', 'encoder.layer.6.intermediate.dense.weight', 'encoder.layer.6.output.LayerNorm.bias', 'encoder.layer.6.output.LayerNorm.weight', 'encoder.layer.6.output.dense.bias', 'encoder.layer.6.output.dense.weight', 'encoder.layer.7.attention.output.LayerNorm.bias', 'encoder.layer.7.attention.output.LayerNorm.weight', 'encoder.layer.7.attention.output.dense.bias', 'encoder.layer.7.attention.output.dense.weight', 'encoder.layer.7.attention.self.key.bias', 'encoder.layer.7.attention.self.key.weight', 'encoder.layer.7.attention.self.query.bias', 'encoder.layer.7.attention.self.query.weight', 'encoder.layer.7.attention.self.value.bias', 'encoder.layer.7.attention.self.value.weight', 'encoder.layer.7.intermediate.dense.bias', 'encoder.layer.7.intermediate.dense.weight', 'encoder.layer.7.output.LayerNorm.bias', 'encoder.layer.7.output.LayerNorm.weight', 'encoder.layer.7.output.dense.bias', 'encoder.layer.7.output.dense.weight', 'encoder.layer.8.attention.output.LayerNorm.bias', 'encoder.layer.8.attention.output.LayerNorm.weight', 'encoder.layer.8.attention.output.dense.bias', 'encoder.layer.8.attention.output.dense.weight', 'encoder.layer.8.attention.self.key.bias', 'encoder.layer.8.attention.self.key.weight', 'encoder.layer.8.attention.self.query.bias', 'encoder.layer.8.attention.self.query.weight', 'encoder.layer.8.attention.self.value.bias', 'encoder.layer.8.attention.self.value.weight', 'encoder.layer.8.intermediate.dense.bias', 'encoder.layer.8.intermediate.dense.weight', 'encoder.layer.8.output.LayerNorm.bias', 'encoder.layer.8.output.LayerNorm.weight', 'encoder.layer.8.output.dense.bias', 'encoder.layer.8.output.dense.weight', 'encoder.layer.9.attention.output.LayerNorm.bias', 'encoder.layer.9.attention.output.LayerNorm.weight', 'encoder.layer.9.attention.output.dense.bias', 'encoder.layer.9.attention.output.dense.weight', 'encoder.layer.9.attention.self.key.bias', 'encoder.layer.9.attention.self.key.weight', 'encoder.layer.9.attention.self.query.bias', 'encoder.layer.9.attention.self.query.weight', 'encoder.layer.9.attention.self.value.bias', 'encoder.layer.9.attention.self.value.weight', 'encoder.layer.9.intermediate.dense.bias', 'encoder.layer.9.intermediate.dense.weight', 'encoder.layer.9.output.LayerNorm.bias', 'encoder.layer.9.output.LayerNorm.weight', 'encoder.layer.9.output.dense.bias', 'encoder.layer.9.output.dense.weight', 'pooler.dense.bias', 'pooler.dense.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Epoch [1/5]
 14%|██████████████████████▌                                                                                                                                        | 200/1407 [00:45<04:36,  4.36it/s]Iter: 200, Train Loss: 0.97, Train Acc: 65.62%, Val Loss: 0.76, Val Acc: 74.78%, Time: 0:00:53 *
 28%|█████████████████████████████████████████████▏                                                                                                                 | 400/1407 [01:39<03:58,  4.22it/s]Iter: 400, Train Loss: 0.75, Train Acc: 75.78%, Val Loss: 0.55, Val Acc: 82.30%, Time: 0:01:48 *
 43%|███████████████████████████████████████████████████████████████████▊                                                                                           | 600/1407 [02:34<03:07,  4.29it/s]Iter: 600, Train Loss: 0.47, Train Acc: 83.59%, Val Loss: 0.53, Val Acc: 82.51%, Time: 0:02:43 *
 57%|██████████████████████████████████████████████████████████████████████████████████████████▍                                                                    | 800/1407 [03:29<02:23,  4.24it/s]Iter: 800, Train Loss: 0.45, Train Acc: 85.16%, Val Loss: 0.47, Val Acc: 85.14%, Time: 0:03:38 *
 71%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                             | 1000/1407 [04:24<01:35,  4.27it/s]Iter: 1000, Train Loss: 0.34, Train Acc: 88.28%, Val Loss: 0.45, Val Acc: 85.51%, Time: 0:04:32 *
 85%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                       | 1200/1407 [05:18<00:48,  4.28it/s]Iter: 1200, Train Loss: 0.49, Train Acc: 83.59%, Val Loss: 0.50, Val Acc: 84.37%, Time: 0:05:26 
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏| 1400/1407 [06:12<00:01,  4.28it/s]Iter: 1400, Train Loss: 0.49, Train Acc: 85.94%, Val Loss: 0.41, Val Acc: 87.25%, Time: 0:06:20 *
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1407/1407 [06:21<00:00,  3.69it/s]
Epoch [2/5]
 14%|█████████████████████▊                                                                                                                                         | 193/1407 [00:44<04:44,  4.27it/s]Iter: 1600, Train Loss: 0.33, Train Acc: 89.06%, Val Loss: 0.47, Val Acc: 85.39%, Time: 0:07:13 
 28%|████████████████████████████████████████████▍                                                                                                                  | 393/1407 [01:38<03:57,  4.27it/s]Iter: 1800, Train Loss: 0.28, Train Acc: 92.97%, Val Loss: 0.37, Val Acc: 88.06%, Time: 0:08:08 *
 42%|███████████████████████████████████████████████████████████████████                                                                                            | 593/1407 [02:32<03:11,  4.26it/s]Iter: 2000, Train Loss: 0.37, Train Acc: 86.72%, Val Loss: 0.38, Val Acc: 88.20%, Time: 0:09:01 
 56%|█████████████████████████████████████████████████████████████████████████████████████████▌                                                                     | 793/1407 [03:26<02:23,  4.28it/s]Iter: 2200, Train Loss: 0.29, Train Acc: 89.06%, Val Loss: 0.37, Val Acc: 88.55%, Time: 0:09:56 *
 71%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                              | 993/1407 [04:20<01:37,  4.24it/s]Iter: 2400, Train Loss: 0.26, Train Acc: 89.84%, Val Loss: 0.41, Val Acc: 87.10%, Time: 0:10:49 
 85%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                        | 1193/1407 [05:14<00:50,  4.23it/s]Iter: 2600, Train Loss: 0.36, Train Acc: 89.84%, Val Loss: 0.36, Val Acc: 88.40%, Time: 0:11:44 *
 99%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍ | 1393/1407 [06:08<00:03,  4.26it/s]Iter: 2800, Train Loss: 0.49, Train Acc: 82.81%, Val Loss: 0.38, Val Acc: 88.38%, Time: 0:12:37 
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1407/1407 [06:18<00:00,  3.72it/s]
Epoch [3/5]
 13%|█████████████████████                                                                                                                                          | 186/1407 [00:43<04:45,  4.28it/s]Iter: 3000, Train Loss: 0.31, Train Acc: 89.06%, Val Loss: 0.35, Val Acc: 88.95%, Time: 0:13:32 *
 27%|███████████████████████████████████████████▌                                                                                                                   | 386/1407 [01:37<03:59,  4.26it/s]Iter: 3200, Train Loss: 0.42, Train Acc: 89.84%, Val Loss: 0.35, Val Acc: 88.89%, Time: 0:14:25 
 42%|██████████████████████████████████████████████████████████████████▏                                                                                            | 586/1407 [02:31<03:12,  4.26it/s]Iter: 3400, Train Loss: 0.40, Train Acc: 88.28%, Val Loss: 0.34, Val Acc: 89.01%, Time: 0:15:20 *
 56%|████████████████████████████████████████████████████████████████████████████████████████▊                                                                      | 786/1407 [03:25<02:24,  4.29it/s]Iter: 3600, Train Loss: 0.21, Train Acc: 92.19%, Val Loss: 0.35, Val Acc: 89.05%, Time: 0:16:13 
 70%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                               | 986/1407 [04:19<01:38,  4.27it/s]Iter: 3800, Train Loss: 0.30, Train Acc: 89.84%, Val Loss: 0.38, Val Acc: 88.02%, Time: 0:17:06 
 84%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                        | 1186/1407 [05:12<00:51,  4.28it/s]Iter: 4000, Train Loss: 0.26, Train Acc: 91.41%, Val Loss: 0.37, Val Acc: 88.54%, Time: 0:18:00 
 99%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋  | 1386/1407 [06:05<00:04,  4.28it/s]Iter: 4200, Train Loss: 0.39, Train Acc: 88.28%, Val Loss: 0.36, Val Acc: 88.93%, Time: 0:18:53 
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1407/1407 [06:17<00:00,  3.73it/s]
Epoch [4/5]
 13%|████████████████████▏                                                                                                                                          | 179/1407 [00:41<04:49,  4.24it/s]Iter: 4400, Train Loss: 0.21, Train Acc: 92.19%, Val Loss: 0.35, Val Acc: 89.26%, Time: 0:19:46 
No optimization for a long time, auto-stopping...
 13%|████████████████████▏                                                                                                                                          | 179/1407 [00:48<05:33,  3.68it/s]
```

预测结果如下：

```css
Test Loss:  0.34, Test Acc: 89.39%
Precision, Recall and F1-Score...
               precision    recall  f1-score   support

      finance     0.8865    0.8670    0.8766      1000
       realty     0.9024    0.9150    0.9086      1000
       stocks     0.8684    0.7920    0.8285      1000
    education     0.9100    0.9610    0.9348      1000
      science     0.8792    0.7640    0.8175      1000
      society     0.9309    0.8750    0.9021      1000
     politics     0.8405    0.9010    0.8697      1000
       sports     0.9358    0.9760    0.9555      1000
         game     0.9058    0.9420    0.9235      1000
entertainment     0.8792    0.9460    0.9114      1000

     accuracy                         0.8939     10000
    macro avg     0.8939    0.8939    0.8928     10000
 weighted avg     0.8939    0.8939    0.8928     10000

Confusion Matrix...
[[867  23  49   7  16   3  17   8   3   7]
 [ 10 915  14   1   5  16   4  20   3  12]
 [ 71  34 792   4  35   0  44   8   7   5]
 [  2   2   0 961   3   5   8   2   4  13]
 [ 10  10  36  27 764  12  36   8  61  36]
 [  2  13   1  25   9 875  45   2   3  25]
 [ 12   7  13  17  10  14 901   4   6  16]
 [  0   2   1   3   0   2   7 976   0   9]
 [  1   4   6   4  23   8   3   2 942   7]
 [  3   4   0   7   4   5   7  13  11 946]]
Time usage: 0:00:06
```

### 5.11 MENGZI-Finetune

#### 5.11.1 MENGZI配置

`mengzi-bert-base`配置如下：

```json
{
  "architectures": [
    "BertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "bos_token_id": 0,
  "directionality": "bidi",
  "eos_token_id": 2,
  "gradient_checkpointing": false,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "output_past": true,
  "pad_token_id": 1,
  "pooler_fc_size": 768,
  "pooler_num_attention_heads": 12,
  "pooler_num_fc_layers": 3,
  "pooler_size_per_head": 128,
  "pooler_type": "first_token_transform",
  "position_embedding_type": "absolute",
  "torch_dtype": "float16",
  "transformers_version": "4.9.2",
  "type_vocab_size": 2,
  "use_cache": true,
  "vocab_size": 21128
}
```

mengzi模型架构如下：

```css
Model(
  (mengzi): BertModel(
    (embeddings): BertEmbeddings(
      (word_embeddings): Embedding(21128, 768, padding_idx=1)
      (position_embeddings): Embedding(512, 768)
      (token_type_embeddings): Embedding(2, 768)
      (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (encoder): BertEncoder(
      (layer): ModuleList(
        (0-11): 12 x BertLayer(
          (attention): BertAttention(
            (self): BertSdpaSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation()
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
    )
    (pooler): BertPooler(
      (dense): Linear(in_features=768, out_features=768, bias=True)
      (activation): Tanh()
    )
  )
  (fc): Linear(in_features=768, out_features=10, bias=True)
)
```

#### 5.11.2 训练和预测

直接调用如下命令：

```shell
python run_mengzi.py --model mengzi
```

训练过程和结果如下：

```css
Loading data for mengzi Model...
180000it [00:30, 5951.78it/s]
10000it [00:01, 6215.40it/s]
10000it [00:01, 6099.53it/s]
Epoch [1/5]
 14%|██████████████████████▌                                                                                                                                        | 200/1407 [00:45<04:32,  4.42it/s]Iter: 200, Train Loss: 0.32, Train Acc: 88.28%, Val Loss: 0.31, Val Acc: 90.73%, Time: 0:00:53 *
 28%|█████████████████████████████████████████████▏                                                                                                                 | 400/1407 [01:38<03:56,  4.26it/s]Iter: 400, Train Loss: 0.43, Train Acc: 85.94%, Val Loss: 0.26, Val Acc: 91.96%, Time: 0:01:47 *
 43%|███████████████████████████████████████████████████████████████████▊                                                                                           | 600/1407 [02:33<03:05,  4.35it/s]Iter: 600, Train Loss: 0.32, Train Acc: 88.28%, Val Loss: 0.25, Val Acc: 92.12%, Time: 0:02:41 *
 57%|██████████████████████████████████████████████████████████████████████████████████████████▍                                                                    | 800/1407 [03:27<02:20,  4.33it/s]Iter: 800, Train Loss: 0.16, Train Acc: 92.97%, Val Loss: 0.22, Val Acc: 93.06%, Time: 0:03:35 *
 71%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                             | 1000/1407 [04:20<01:34,  4.29it/s]Iter: 1000, Train Loss: 0.17, Train Acc: 93.75%, Val Loss: 0.22, Val Acc: 92.90%, Time: 0:04:29 *
 85%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                       | 1200/1407 [05:14<00:47,  4.33it/s]Iter: 1200, Train Loss: 0.20, Train Acc: 92.97%, Val Loss: 0.21, Val Acc: 93.00%, Time: 0:05:23 *
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏| 1400/1407 [06:08<00:01,  4.32it/s]Iter: 1400, Train Loss: 0.28, Train Acc: 91.41%, Val Loss: 0.20, Val Acc: 93.48%, Time: 0:06:17 *
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1407/1407 [06:17<00:00,  3.72it/s]
Epoch [2/5]
 14%|█████████████████████▊                                                                                                                                         | 193/1407 [00:44<04:40,  4.33it/s]Iter: 1600, Train Loss: 0.23, Train Acc: 91.41%, Val Loss: 0.20, Val Acc: 93.44%, Time: 0:07:09 
 28%|████████████████████████████████████████████▍                                                                                                                  | 393/1407 [01:37<03:54,  4.32it/s]Iter: 1800, Train Loss: 0.16, Train Acc: 95.31%, Val Loss: 0.20, Val Acc: 93.60%, Time: 0:08:03 *
 42%|███████████████████████████████████████████████████████████████████                                                                                            | 593/1407 [02:30<03:08,  4.32it/s]Iter: 2000, Train Loss: 0.16, Train Acc: 96.09%, Val Loss: 0.20, Val Acc: 93.67%, Time: 0:08:57 *
 56%|█████████████████████████████████████████████████████████████████████████████████████████▌                                                                     | 793/1407 [03:24<02:21,  4.33it/s]Iter: 2200, Train Loss: 0.17, Train Acc: 94.53%, Val Loss: 0.20, Val Acc: 93.85%, Time: 0:09:51 *
 71%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                              | 993/1407 [04:18<01:36,  4.31it/s]Iter: 2400, Train Loss: 0.10, Train Acc: 96.09%, Val Loss: 0.20, Val Acc: 93.59%, Time: 0:10:44 
 85%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                        | 1193/1407 [05:11<00:49,  4.29it/s]Iter: 2600, Train Loss: 0.14, Train Acc: 94.53%, Val Loss: 0.19, Val Acc: 93.84%, Time: 0:11:38 *
 99%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍ | 1393/1407 [06:05<00:03,  4.32it/s]Iter: 2800, Train Loss: 0.11, Train Acc: 96.88%, Val Loss: 0.19, Val Acc: 93.75%, Time: 0:12:30 
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1407/1407 [06:15<00:00,  3.75it/s]
Epoch [3/5]
 13%|█████████████████████                                                                                                                                          | 186/1407 [00:42<04:41,  4.34it/s]Iter: 3000, Train Loss: 0.11, Train Acc: 97.66%, Val Loss: 0.20, Val Acc: 93.91%, Time: 0:13:23 
 27%|███████████████████████████████████████████▌                                                                                                                   | 386/1407 [01:35<03:55,  4.33it/s]Iter: 3200, Train Loss: 0.16, Train Acc: 95.31%, Val Loss: 0.20, Val Acc: 93.87%, Time: 0:14:15 
 42%|██████████████████████████████████████████████████████████████████▏                                                                                            | 586/1407 [02:28<03:09,  4.32it/s]Iter: 3400, Train Loss: 0.17, Train Acc: 96.88%, Val Loss: 0.20, Val Acc: 93.92%, Time: 0:15:08 
 56%|████████████████████████████████████████████████████████████████████████████████████████▊                                                                      | 786/1407 [03:20<02:23,  4.33it/s]Iter: 3600, Train Loss: 0.07, Train Acc: 96.88%, Val Loss: 0.20, Val Acc: 93.85%, Time: 0:16:01 
No optimization for a long time, auto-stopping...
 56%|████████████████████████████████████████████████████████████████████████████████████████▊                                                                      | 786/1407 [03:27<02:44,  3.78it/s]
Test Loss:  0.18, Test Acc: 94.69%
Precision, Recall and F1-Score...
               precision    recall  f1-score   support

      finance     0.9237    0.9560    0.9396      1000
       realty     0.9513    0.9570    0.9541      1000
       stocks     0.9343    0.8820    0.9074      1000
    education     0.9680    0.9690    0.9685      1000
      science     0.9196    0.9260    0.9228      1000
      society     0.9408    0.9380    0.9394      1000
     politics     0.9302    0.9200    0.9251      1000
       sports     0.9850    0.9880    0.9865      1000
         game     0.9688    0.9640    0.9664      1000
entertainment     0.9472    0.9690    0.9580      1000

     accuracy                         0.9469     10000
    macro avg     0.9469    0.9469    0.9468     10000
 weighted avg     0.9469    0.9469    0.9468     10000

Confusion Matrix...
[[956   9  20   1   5   2   6   1   0   0]
 [ 14 957   4   2   1   4   6   2   2   8]
 [ 49  19 882   0  23   5  20   0   0   2]
 [  3   0   2 969   4   4   6   1   2   9]
 [  2   3  13   3 926  14   7   0  21  11]
 [  0   9   2  13   5 938  17   0   4  12]
 [  7   6  16   7  15  23 920   2   0   4]
 [  2   1   2   1   1   0   2 988   0   3]
 [  0   0   2   0  25   2   1   1 964   5]
 [  2   2   1   5   2   5   4   8   2 969]]
Time usage: 0:00:06
```

### 5.12 NEZHA-Finetune

#### 5.12.1 NEZHA配置

`nezha-cn-base`配置如下：

```json
{
  "_name_or_path": "nezha-cn-base",
  "architectures": [
    "NeZhaForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "bos_token_id": 2,
  "classifier_dropout": 0.1,
  "embedding_size": 128,
  "eos_token_id": 3,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "inner_group_num": 1,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "max_relative_position": 64,
  "model_type": "nezha",
  "num_attention_heads": 12,
  "num_hidden_groups": 1,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "torch_dtype": "float32",
  "transformers_version": "4.20.0.dev0",
  "type_vocab_size": 2,
  "use_cache": true,
  "use_relative_position": true,
  "vocab_size": 21128
}
```

NEZHA模型架构如下：

```css
Model(
  (nezha): NezhaModel(
    (embeddings): NezhaEmbeddings(
      (word_embeddings): Embedding(21128, 768, padding_idx=0)
      (token_type_embeddings): Embedding(2, 768)
      (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (encoder): NezhaEncoder(
      (layer): ModuleList(
        (0-11): 12 x NezhaLayer(
          (attention): NezhaAttention(
            (self): NezhaSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
              (relative_positions_encoding): NezhaRelativePositionsEncoding()
            )
            (output): NezhaSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (intermediate): NezhaIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation()
          )
          (output): NezhaOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
    )
    (pooler): NezhaPooler(
      (dense): Linear(in_features=768, out_features=768, bias=True)
      (activation): Tanh()
    )
  )
  (fc): Linear(in_features=768, out_features=10, bias=True)
)
```

#### 5.12.2 训练和预测

直接调用如下命令：

```shell
python run_nezha.py --model nezha
```

训练过程和结果如下：

```css
Loading data for nezha Model...
180000it [00:29, 6159.58it/s]
10000it [00:01, 6488.68it/s]
10000it [00:01, 6158.05it/s]
Epoch [1/5]
 14%|███████████████████████▍                                                                                                                                             | 200/1407 [00:48<04:42,  4.28it/s]Iter: 200, Train Loss: 0.36, Train Acc: 89.06%, Val Loss: 0.32, Val Acc: 91.76%, Time: 0:00:56 *
 28%|██████████████████████████████████████████████▉                                                                                                                      | 400/1407 [01:42<03:57,  4.24it/s]Iter: 400, Train Loss: 0.37, Train Acc: 89.84%, Val Loss: 0.27, Val Acc: 92.70%, Time: 0:01:50 *
 43%|██████████████████████████████████████████████████████████████████████▎                                                                                              | 600/1407 [02:36<03:10,  4.23it/s]Iter: 600, Train Loss: 0.29, Train Acc: 89.84%, Val Loss: 0.24, Val Acc: 93.30%, Time: 0:02:44 *
 57%|█████████████████████████████████████████████████████████████████████████████████████████████▊                                                                       | 800/1407 [03:31<02:23,  4.23it/s]Iter: 800, Train Loss: 0.16, Train Acc: 95.31%, Val Loss: 0.23, Val Acc: 93.31%, Time: 0:03:39 *
 71%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                               | 1000/1407 [04:25<01:36,  4.24it/s]Iter: 1000, Train Loss: 0.22, Train Acc: 90.62%, Val Loss: 0.23, Val Acc: 93.29%, Time: 0:04:33 *
 85%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                        | 1200/1407 [05:19<00:48,  4.23it/s]Iter: 1200, Train Loss: 0.18, Train Acc: 94.53%, Val Loss: 0.21, Val Acc: 93.53%, Time: 0:05:27 *
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏| 1400/1407 [06:13<00:01,  4.24it/s]Iter: 1400, Train Loss: 0.26, Train Acc: 92.97%, Val Loss: 0.20, Val Acc: 94.02%, Time: 0:06:21 *
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1407/1407 [06:21<00:00,  3.68it/s]
Epoch [2/5]
 14%|██████████████████████▋                                                                                                                                              | 193/1407 [00:45<04:47,  4.23it/s]Iter: 1600, Train Loss: 0.20, Train Acc: 93.75%, Val Loss: 0.20, Val Acc: 93.95%, Time: 0:07:14 
 28%|██████████████████████████████████████████████                                                                                                                       | 393/1407 [01:38<03:59,  4.24it/s]Iter: 1800, Train Loss: 0.18, Train Acc: 95.31%, Val Loss: 0.21, Val Acc: 93.75%, Time: 0:08:07 
 42%|█████████████████████████████████████████████████████████████████████▌                                                                                               | 593/1407 [02:32<03:12,  4.23it/s]Iter: 2000, Train Loss: 0.15, Train Acc: 94.53%, Val Loss: 0.19, Val Acc: 94.15%, Time: 0:09:02 *
 56%|████████████████████████████████████████████████████████████████████████████████████████████▉                                                                        | 793/1407 [03:26<02:25,  4.23it/s]Iter: 2200, Train Loss: 0.19, Train Acc: 94.53%, Val Loss: 0.20, Val Acc: 94.03%, Time: 0:09:55 
 71%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                                | 993/1407 [04:19<01:37,  4.23it/s]Iter: 2400, Train Loss: 0.07, Train Acc: 97.66%, Val Loss: 0.19, Val Acc: 94.30%, Time: 0:10:48 
 85%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                         | 1193/1407 [05:13<00:50,  4.23it/s]Iter: 2600, Train Loss: 0.13, Train Acc: 96.88%, Val Loss: 0.19, Val Acc: 94.14%, Time: 0:11:42 
 99%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎ | 1393/1407 [06:06<00:03,  4.23it/s]Iter: 2800, Train Loss: 0.08, Train Acc: 98.44%, Val Loss: 0.19, Val Acc: 94.08%, Time: 0:12:35 
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1407/1407 [06:16<00:00,  3.74it/s]
Epoch [3/5]
 13%|█████████████████████▊                                                                                                                                               | 186/1407 [00:43<04:48,  4.23it/s]Iter: 3000, Train Loss: 0.09, Train Acc: 98.44%, Val Loss: 0.20, Val Acc: 93.95%, Time: 0:13:28 
No optimization for a long time, auto-stopping...
 13%|█████████████████████▊                                                                                                                                               | 186/1407 [00:50<05:30,  3.69it/s]
```

预测结果如下：

```css
Test Loss:   0.2, Test Acc: 94.15%
Precision, Recall and F1-Score...
               precision    recall  f1-score   support

      finance     0.9343    0.9390    0.9367      1000
       realty     0.9390    0.9700    0.9543      1000
       stocks     0.9508    0.8310    0.8869      1000
    education     0.9641    0.9670    0.9656      1000
      science     0.8860    0.9400    0.9122      1000
      society     0.9142    0.9590    0.9361      1000
     politics     0.9064    0.9390    0.9224      1000
       sports     0.9898    0.9750    0.9824      1000
         game     0.9851    0.9270    0.9552      1000
entertainment     0.9556    0.9680    0.9617      1000

     accuracy                         0.9415     10000
    macro avg     0.9425    0.9415    0.9413     10000
 weighted avg     0.9425    0.9415    0.9413     10000

Confusion Matrix...
[[939  12  18   1   7  11   9   1   1   1]
 [  6 970   3   4   2   7   4   0   1   3]
 [ 50  31 831   0  39   3  43   1   0   2]
 [  2   2   0 967   2  15   8   0   1   3]
 [  1   3   7   5 940  14  15   0   5  10]
 [  0   5   1  11   4 959  11   0   0   9]
 [  2   4  11   8  10  20 939   1   0   5]
 [  1   3   1   1   1   6   3 975   0   9]
 [  3   1   2   1  50  10   3   0 927   3]
 [  1   2   0   5   6   4   1   7   6 968]]
Time usage: 0:00:06
```

## 6. 知识蒸馏

### 6.1 模型使用和配置

TextCNN 初始模型架构如下(词表大小在真实训练时会变化):

```css
Model(
  (embedding): Embedding(100, 300, padding_idx=99)
  (convs): ModuleList(
    (0): Conv2d(1, 512, kernel_size=(2, 300), stride=(1, 1))
    (1): Conv2d(1, 512, kernel_size=(3, 300), stride=(1, 1))
    (2): Conv2d(1, 512, kernel_size=(4, 300), stride=(1, 1))
  )
  (dropout): Dropout(p=0.5, inplace=False)
  (fc): Linear(in_features=1536, out_features=10, bias=True)
)
```

### 6.2 模型训练和预测

直接执行如下命令进行训练

```bash
python run_textCNN.py --task train_kd
```

训练过程如下：

```css
180000it [00:28, 6229.26it/s]
10000it [00:01, 6360.99it/s]
10000it [00:01, 6416.83it/s]
180000it [00:01, 122049.20it/s]
10000it [00:00, 108980.33it/s]
10000it [00:00, 153076.79it/s]
Data loaded, now load teacher model
Teacher and student models loaded, start training
Epoch [1/50]
  0%|                                                                                                                                                                         | 0/1407 [00:00<?, ?it/s]Iter:      0,  Train Loss:   5.4,  Train Acc:  9.38%,  Val Loss:   2.6,  Val Acc: 15.18%,  Time: 0:01:40 *
  7%|███████████▏                                                                                                                                                    | 98/1407 [00:02<00:13, 94.10it/s]Iter:    100,  Train Loss:   1.7,  Train Acc: 75.78%,  Val Loss:  0.69,  Val Acc: 78.28%,  Time: 0:01:41 *
 14%|█████████████████████▊                                                                                                                                        | 194/1407 [00:03<00:11, 108.92it/s]Iter:    200,  Train Loss:   1.5,  Train Acc: 78.12%,  Val Loss:  0.55,  Val Acc: 82.58%,  Time: 0:01:43 *
 21%|████████████████████████████████▌                                                                                                                             | 290/1407 [00:04<00:10, 110.16it/s]Iter:    300,  Train Loss:   1.2,  Train Acc: 79.69%,  Val Loss:  0.52,  Val Acc: 83.53%,  Time: 0:01:44 *
 28%|████████████████████████████████████████████▊                                                                                                                 | 399/1407 [00:05<00:09, 110.89it/s]Iter:    400,  Train Loss:   1.4,  Train Acc: 75.00%,  Val Loss:  0.47,  Val Acc: 85.71%,  Time: 0:01:45 *
 35%|███████████████████████████████████████████████████████▌                                                                                                      | 495/1407 [00:06<00:08, 109.35it/s]Iter:    500,  Train Loss:   1.2,  Train Acc: 82.03%,  Val Loss:  0.45,  Val Acc: 85.78%,  Time: 0:01:46 *
 42%|██████████████████████████████████████████████████████████████████▎                                                                                           | 591/1407 [00:07<00:07, 108.23it/s]Iter:    600,  Train Loss:   1.0,  Train Acc: 83.59%,  Val Loss:  0.43,  Val Acc: 86.43%,  Time: 0:01:47 *
 50%|██████████████████████████████████████████████████████████████████████████████▌                                                                               | 700/1407 [00:08<00:06, 110.81it/s]Iter:    700,  Train Loss:   1.2,  Train Acc: 82.03%,  Val Loss:  0.45,  Val Acc: 86.02%,  Time: 0:01:48 
 57%|█████████████████████████████████████████████████████████████████████████████████████████▍                                                                    | 796/1407 [00:09<00:05, 109.78it/s]Iter:    800,  Train Loss:   1.2,  Train Acc: 78.91%,  Val Loss:  0.42,  Val Acc: 86.92%,  Time: 0:01:49 *
 63%|████████████████████████████████████████████████████████████████████████████████████████████████████▏                                                         | 892/1407 [00:10<00:04, 109.76it/s]Iter:    900,  Train Loss:  0.93,  Train Acc: 87.50%,  Val Loss:   0.4,  Val Acc: 87.86%,  Time: 0:01:50 *
 71%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                             | 1000/1407 [00:12<00:03, 110.87it/s]Iter:   1000,  Train Loss:  0.84,  Train Acc: 88.28%,  Val Loss:   0.4,  Val Acc: 87.27%,  Time: 0:01:51 
 78%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                  | 1096/1407 [00:13<00:02, 109.47it/s]Iter:   1100,  Train Loss:  0.87,  Train Acc: 89.06%,  Val Loss:  0.42,  Val Acc: 87.23%,  Time: 0:01:52 
 85%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                        | 1192/1407 [00:14<00:01, 109.26it/s]Iter:   1200,  Train Loss:  0.92,  Train Acc: 82.81%,  Val Loss:  0.39,  Val Acc: 87.94%,  Time: 0:01:54 *
 92%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████            | 1300/1407 [00:15<00:00, 108.96it/s]Iter:   1300,  Train Loss:   0.9,  Train Acc: 84.38%,  Val Loss:  0.38,  Val Acc: 88.12%,  Time: 0:01:55 *
 99%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊ | 1396/1407 [00:16<00:00, 95.90it/s]Iter:   1400,  Train Loss:  0.96,  Train Acc: 77.34%,  Val Loss:  0.38,  Val Acc: 88.08%,  Time: 0:01:56 *
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1407/1407 [00:17<00:00, 81.18it/s]
Epoch [2/50]
  6%|█████████▋                                                                                                                                                     | 86/1407 [00:00<00:11, 116.29it/s]Iter:   1500,  Train Loss:  0.92,  Train Acc: 85.94%,  Val Loss:  0.39,  Val Acc: 87.93%,  Time: 0:01:57 
 13%|████████████████████▍                                                                                                                                         | 182/1407 [00:01<00:11, 110.42it/s]Iter:   1600,  Train Loss:  0.68,  Train Acc: 89.84%,  Val Loss:  0.36,  Val Acc: 88.70%,  Time: 0:01:58 *
 21%|████████████████████████████████▋                                                                                                                             | 291/1407 [00:02<00:10, 111.34it/s]Iter:   1700,  Train Loss:  0.79,  Train Acc: 85.94%,  Val Loss:  0.38,  Val Acc: 88.45%,  Time: 0:01:59 
 28%|███████████████████████████████████████████▍                                                                                                                  | 387/1407 [00:03<00:09, 109.92it/s]Iter:   1800,  Train Loss:   0.6,  Train Acc: 88.28%,  Val Loss:  0.35,  Val Acc: 89.20%,  Time: 0:02:01 *
 34%|██████████████████████████████████████████████████████▏                                                                                                       | 483/1407 [00:05<00:08, 106.20it/s]Iter:   1900,  Train Loss:  0.74,  Train Acc: 86.72%,  Val Loss:  0.35,  Val Acc: 88.93%,  Time: 0:02:02 *
 42%|██████████████████████████████████████████████████████████████████▍                                                                                           | 592/1407 [00:06<00:07, 110.72it/s]Iter:   2000,  Train Loss:  0.87,  Train Acc: 87.50%,  Val Loss:  0.35,  Val Acc: 89.11%,  Time: 0:02:03 
 49%|█████████████████████████████████████████████████████████████████████████████▎                                                                                | 688/1407 [00:07<00:06, 109.07it/s]Iter:   2100,  Train Loss:  0.81,  Train Acc: 85.16%,  Val Loss:  0.34,  Val Acc: 89.26%,  Time: 0:02:04 *
 56%|████████████████████████████████████████████████████████████████████████████████████████                                                                      | 784/1407 [00:08<00:05, 109.55it/s]Iter:   2200,  Train Loss:  0.72,  Train Acc: 87.50%,  Val Loss:  0.35,  Val Acc: 89.44%,  Time: 0:02:05 
 63%|████████████████████████████████████████████████████████████████████████████████████████████████████▏                                                         | 892/1407 [00:09<00:04, 110.15it/s]Iter:   2300,  Train Loss:  0.68,  Train Acc: 92.19%,  Val Loss:  0.35,  Val Acc: 89.31%,  Time: 0:02:06 
 70%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                                               | 988/1407 [00:10<00:03, 109.16it/s]Iter:   2400,  Train Loss:  0.68,  Train Acc: 92.19%,  Val Loss:  0.35,  Val Acc: 89.50%,  Time: 0:02:07 
 77%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                                    | 1084/1407 [00:11<00:02, 108.69it/s]Iter:   2500,  Train Loss:  0.59,  Train Acc: 91.41%,  Val Loss:  0.35,  Val Acc: 89.11%,  Time: 0:02:08 
 85%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                        | 1192/1407 [00:12<00:01, 109.93it/s]Iter:   2600,  Train Loss:  0.61,  Train Acc: 89.06%,  Val Loss:  0.35,  Val Acc: 89.52%,  Time: 0:02:09 
 92%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋             | 1288/1407 [00:13<00:01, 108.96it/s]Iter:   2700,  Train Loss:  0.67,  Train Acc: 89.06%,  Val Loss:  0.34,  Val Acc: 89.43%,  Time: 0:02:10 *
 98%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍  | 1384/1407 [00:14<00:00, 108.50it/s]Iter:   2800,  Train Loss:  0.78,  Train Acc: 84.38%,  Val Loss:  0.33,  Val Acc: 89.58%,  Time: 0:02:11 *
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1407/1407 [00:15<00:00, 92.82it/s]
Epoch [3/50]
  6%|█████████▋                                                                                                                                                     | 86/1407 [00:00<00:11, 114.98it/s]Iter:   2900,  Train Loss:  0.55,  Train Acc: 90.62%,  Val Loss:  0.33,  Val Acc: 89.35%,  Time: 0:02:12 
 13%|████████████████████▍                                                                                                                                         | 182/1407 [00:01<00:11, 109.25it/s]Iter:   3000,  Train Loss:  0.62,  Train Acc: 89.06%,  Val Loss:  0.32,  Val Acc: 90.10%,  Time: 0:02:14 *
 20%|███████████████████████████████▏                                                                                                                              | 278/1407 [00:02<00:10, 108.99it/s]Iter:   3100,  Train Loss:  0.59,  Train Acc: 92.97%,  Val Loss:  0.32,  Val Acc: 89.92%,  Time: 0:02:15 *
 27%|███████████████████████████████████████████▎                                                                                                                  | 386/1407 [00:03<00:09, 110.42it/s]Iter:   3200,  Train Loss:  0.55,  Train Acc: 91.41%,  Val Loss:  0.33,  Val Acc: 89.57%,  Time: 0:02:16 
 34%|██████████████████████████████████████████████████████▏                                                                                                       | 482/1407 [00:04<00:08, 108.88it/s]Iter:   3300,  Train Loss:  0.63,  Train Acc: 89.06%,  Val Loss:  0.33,  Val Acc: 89.89%,  Time: 0:02:17 
 41%|████████████████████████████████████████████████████████████████▉                                                                                             | 578/1407 [00:05<00:07, 109.07it/s]Iter:   3400,  Train Loss:  0.64,  Train Acc: 89.84%,  Val Loss:  0.32,  Val Acc: 90.00%,  Time: 0:02:18 *
 49%|█████████████████████████████████████████████████████████████████████████████                                                                                 | 686/1407 [00:07<00:06, 110.53it/s]Iter:   3500,  Train Loss:  0.59,  Train Acc: 90.62%,  Val Loss:  0.32,  Val Acc: 89.98%,  Time: 0:02:19 *
 56%|███████████████████████████████████████████████████████████████████████████████████████▊                                                                      | 782/1407 [00:08<00:05, 107.58it/s]Iter:   3600,  Train Loss:  0.55,  Train Acc: 94.53%,  Val Loss:  0.32,  Val Acc: 89.94%,  Time: 0:02:20 
 62%|██████████████████████████████████████████████████████████████████████████████████████████████████▌                                                           | 878/1407 [00:09<00:04, 109.50it/s]Iter:   3700,  Train Loss:  0.56,  Train Acc: 89.84%,  Val Loss:  0.32,  Val Acc: 90.04%,  Time: 0:02:21 
 70%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                               | 986/1407 [00:10<00:03, 110.29it/s]Iter:   3800,  Train Loss:  0.66,  Train Acc: 92.19%,  Val Loss:  0.32,  Val Acc: 90.33%,  Time: 0:02:22 *
 77%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                    | 1082/1407 [00:11<00:03, 108.15it/s]Iter:   3900,  Train Loss:  0.62,  Train Acc: 89.06%,  Val Loss:  0.32,  Val Acc: 90.24%,  Time: 0:02:23 
 84%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                         | 1178/1407 [00:12<00:02, 109.88it/s]Iter:   4000,  Train Loss:  0.61,  Train Acc: 90.62%,  Val Loss:  0.32,  Val Acc: 90.39%,  Time: 0:02:24 
 91%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍             | 1286/1407 [00:13<00:01, 110.40it/s]Iter:   4100,  Train Loss:  0.59,  Train Acc: 92.19%,  Val Loss:  0.31,  Val Acc: 90.17%,  Time: 0:02:25 *
 98%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏  | 1382/1407 [00:14<00:00, 109.33it/s]Iter:   4200,  Train Loss:  0.48,  Train Acc: 90.62%,  Val Loss:  0.32,  Val Acc: 89.80%,  Time: 0:02:26 
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1407/1407 [00:15<00:00, 93.69it/s]
Epoch [4/50]
  5%|████████▎                                                                                                                                                      | 74/1407 [00:00<00:11, 114.54it/s]Iter:   4300,  Train Loss:  0.67,  Train Acc: 89.84%,  Val Loss:  0.31,  Val Acc: 90.61%,  Time: 0:02:27 *
 12%|███████████████████                                                                                                                                           | 170/1407 [00:01<00:11, 109.95it/s]Iter:   4400,  Train Loss:  0.45,  Train Acc: 96.88%,  Val Loss:   0.3,  Val Acc: 90.34%,  Time: 0:02:29 *
 20%|███████████████████████████████▎                                                                                                                              | 279/1407 [00:02<00:10, 110.99it/s]Iter:   4500,  Train Loss:  0.51,  Train Acc: 89.84%,  Val Loss:  0.31,  Val Acc: 90.26%,  Time: 0:02:30 
 27%|██████████████████████████████████████████                                                                                                                    | 375/1407 [00:03<00:09, 109.92it/s]Iter:   4600,  Train Loss:  0.51,  Train Acc: 92.19%,  Val Loss:  0.31,  Val Acc: 90.28%,  Time: 0:02:31 
 33%|████████████████████████████████████████████████████▉                                                                                                         | 471/1407 [00:04<00:08, 109.75it/s]Iter:   4700,  Train Loss:  0.48,  Train Acc: 89.84%,  Val Loss:   0.3,  Val Acc: 90.43%,  Time: 0:02:32 *
 41%|█████████████████████████████████████████████████████████████████                                                                                             | 579/1407 [00:06<00:07, 110.77it/s]Iter:   4800,  Train Loss:  0.53,  Train Acc: 92.97%,  Val Loss:   0.3,  Val Acc: 90.65%,  Time: 0:02:33 
 48%|███████████████████████████████████████████████████████████████████████████▊                                                                                  | 675/1407 [00:07<00:06, 109.63it/s]Iter:   4900,  Train Loss:  0.52,  Train Acc: 93.75%,  Val Loss:  0.31,  Val Acc: 90.25%,  Time: 0:02:34 
 55%|██████████████████████████████████████████████████████████████████████████████████████▌                                                                       | 771/1407 [00:08<00:05, 110.03it/s]Iter:   5000,  Train Loss:  0.54,  Train Acc: 92.19%,  Val Loss:  0.32,  Val Acc: 89.82%,  Time: 0:02:35 
 62%|██████████████████████████████████████████████████████████████████████████████████████████████████▋                                                           | 879/1407 [00:09<00:04, 110.91it/s]Iter:   5100,  Train Loss:   0.5,  Train Acc: 92.19%,  Val Loss:   0.3,  Val Acc: 90.74%,  Time: 0:02:36 *
 69%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                                | 975/1407 [00:10<00:03, 110.29it/s]Iter:   5200,  Train Loss:  0.57,  Train Acc: 89.84%,  Val Loss:  0.31,  Val Acc: 90.55%,  Time: 0:02:37 
 76%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                     | 1071/1407 [00:11<00:03, 110.28it/s]Iter:   5300,  Train Loss:  0.47,  Train Acc: 91.41%,  Val Loss:  0.31,  Val Acc: 90.66%,  Time: 0:02:38 
 84%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                         | 1179/1407 [00:12<00:02, 110.71it/s]Iter:   5400,  Train Loss:  0.56,  Train Acc: 90.62%,  Val Loss:  0.31,  Val Acc: 90.25%,  Time: 0:02:39 
 91%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎              | 1275/1407 [00:13<00:01, 109.52it/s]Iter:   5500,  Train Loss:  0.49,  Train Acc: 89.06%,  Val Loss:   0.3,  Val Acc: 90.71%,  Time: 0:02:40 
 97%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉    | 1371/1407 [00:14<00:00, 109.82it/s]Iter:   5600,  Train Loss:  0.42,  Train Acc: 95.31%,  Val Loss:   0.3,  Val Acc: 90.36%,  Time: 0:02:41 
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1407/1407 [00:14<00:00, 95.58it/s]
Epoch [5/50]
  4%|███████                                                                                                                                                        | 62/1407 [00:00<00:11, 114.42it/s]Iter:   5700,  Train Loss:  0.45,  Train Acc: 91.41%,  Val Loss:   0.3,  Val Acc: 90.62%,  Time: 0:02:42 
 12%|███████████████████                                                                                                                                           | 170/1407 [00:01<00:11, 111.45it/s]Iter:   5800,  Train Loss:  0.48,  Train Acc: 91.41%,  Val Loss:   0.3,  Val Acc: 90.58%,  Time: 0:02:43 
 19%|█████████████████████████████▊                                                                                                                                | 266/1407 [00:02<00:10, 109.82it/s]Iter:   5900,  Train Loss:  0.41,  Train Acc: 94.53%,  Val Loss:   0.3,  Val Acc: 90.91%,  Time: 0:02:44 *
 26%|████████████████████████████████████████▋                                                                                                                     | 362/1407 [00:03<00:09, 110.52it/s]Iter:   6000,  Train Loss:  0.45,  Train Acc: 91.41%,  Val Loss:   0.3,  Val Acc: 90.53%,  Time: 0:02:45 
 33%|████████████████████████████████████████████████████▊                                                                                                         | 470/1407 [00:04<00:08, 111.23it/s]Iter:   6100,  Train Loss:  0.57,  Train Acc: 91.41%,  Val Loss:   0.3,  Val Acc: 90.71%,  Time: 0:02:46 
 40%|███████████████████████████████████████████████████████████████▌                                                                                              | 566/1407 [00:05<00:07, 110.19it/s]Iter:   6200,  Train Loss:  0.42,  Train Acc: 97.66%,  Val Loss:   0.3,  Val Acc: 90.54%,  Time: 0:02:47 
 47%|██████████████████████████████████████████████████████████████████████████▎                                                                                   | 662/1407 [00:06<00:06, 110.25it/s]Iter:   6300,  Train Loss:  0.43,  Train Acc: 96.09%,  Val Loss:  0.29,  Val Acc: 90.75%,  Time: 0:02:48 *
 55%|██████████████████████████████████████████████████████████████████████████████████████▌                                                                       | 771/1407 [00:07<00:05, 112.13it/s]Iter:   6400,  Train Loss:  0.38,  Train Acc: 96.09%,  Val Loss:   0.3,  Val Acc: 90.68%,  Time: 0:02:49 
 62%|█████████████████████████████████████████████████████████████████████████████████████████████████▎                                                            | 867/1407 [00:08<00:04, 110.64it/s]Iter:   6500,  Train Loss:  0.44,  Train Acc: 90.62%,  Val Loss:   0.3,  Val Acc: 90.84%,  Time: 0:02:50 
 68%|████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                                 | 963/1407 [00:09<00:04, 110.61it/s]Iter:   6600,  Train Loss:  0.49,  Train Acc: 91.41%,  Val Loss:   0.3,  Val Acc: 90.91%,  Time: 0:02:51 
 76%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                     | 1071/1407 [00:10<00:03, 111.32it/s]Iter:   6700,  Train Loss:  0.42,  Train Acc: 93.75%,  Val Loss:   0.3,  Val Acc: 90.77%,  Time: 0:02:52 
 83%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                           | 1166/1407 [00:11<00:02, 109.64it/s]Iter:   6800,  Train Loss:  0.46,  Train Acc: 94.53%,  Val Loss:   0.3,  Val Acc: 90.56%,  Time: 0:02:53 
 90%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                | 1262/1407 [00:12<00:01, 110.28it/s]Iter:   6900,  Train Loss:  0.39,  Train Acc: 94.53%,  Val Loss:  0.29,  Val Acc: 91.20%,  Time: 0:02:55 *
 97%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉    | 1371/1407 [00:14<00:00, 112.37it/s]Iter:   7000,  Train Loss:  0.45,  Train Acc: 91.41%,  Val Loss:   0.3,  Val Acc: 90.67%,  Time: 0:02:56 
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1407/1407 [00:14<00:00, 96.67it/s]
Epoch [6/50]
  4%|███████                                                                                                                                                        | 62/1407 [00:00<00:11, 115.24it/s]Iter:   7100,  Train Loss:  0.46,  Train Acc: 91.41%,  Val Loss:  0.29,  Val Acc: 90.79%,  Time: 0:02:57 *
 11%|█████████████████▋                                                                                                                                            | 158/1407 [00:01<00:11, 110.25it/s]Iter:   7200,  Train Loss:  0.48,  Train Acc: 89.06%,  Val Loss:  0.29,  Val Acc: 90.71%,  Time: 0:02:58 *
 18%|████████████████████████████▌                                                                                                                                 | 254/1407 [00:02<00:10, 110.45it/s]Iter:   7300,  Train Loss:  0.56,  Train Acc: 93.75%,  Val Loss:  0.29,  Val Acc: 90.88%,  Time: 0:02:59 *
 26%|████████████████████████████████████████▉                                                                                                                     | 364/1407 [00:04<00:09, 108.30it/s]Iter:   7400,  Train Loss:  0.45,  Train Acc: 92.97%,  Val Loss:  0.29,  Val Acc: 91.08%,  Time: 0:03:00 *
 33%|███████████████████████████████████████████████████▋                                                                                                          | 460/1407 [00:05<00:08, 109.57it/s]Iter:   7500,  Train Loss:  0.45,  Train Acc: 93.75%,  Val Loss:  0.29,  Val Acc: 90.91%,  Time: 0:03:01 
 40%|██████████████████████████████████████████████████████████████▍                                                                                               | 556/1407 [00:06<00:07, 111.29it/s]Iter:   7600,  Train Loss:  0.46,  Train Acc: 95.31%,  Val Loss:  0.29,  Val Acc: 90.96%,  Time: 0:03:02 
 47%|██████████████████████████████████████████████████████████████████████████▌                                                                                   | 664/1407 [00:07<00:06, 111.68it/s]Iter:   7700,  Train Loss:  0.48,  Train Acc: 91.41%,  Val Loss:  0.29,  Val Acc: 91.06%,  Time: 0:03:03 
 54%|█████████████████████████████████████████████████████████████████████████████████████▎                                                                        | 760/1407 [00:08<00:05, 110.53it/s]Iter:   7800,  Train Loss:   0.4,  Train Acc: 93.75%,  Val Loss:  0.29,  Val Acc: 91.14%,  Time: 0:03:04 
 61%|████████████████████████████████████████████████████████████████████████████████████████████████▏                                                             | 856/1407 [00:09<00:04, 110.24it/s]Iter:   7900,  Train Loss:  0.46,  Train Acc: 92.97%,  Val Loss:  0.29,  Val Acc: 90.88%,  Time: 0:03:05 
 69%|████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                                 | 964/1407 [00:10<00:03, 111.39it/s]Iter:   8000,  Train Loss:  0.47,  Train Acc: 94.53%,  Val Loss:  0.29,  Val Acc: 91.03%,  Time: 0:03:06 
 75%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                      | 1060/1407 [00:11<00:03, 110.06it/s]Iter:   8100,  Train Loss:  0.38,  Train Acc: 92.19%,  Val Loss:  0.29,  Val Acc: 91.35%,  Time: 0:03:07 
 82%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                            | 1156/1407 [00:12<00:02, 110.29it/s]Iter:   8200,  Train Loss:  0.49,  Train Acc: 91.41%,  Val Loss:  0.29,  Val Acc: 91.11%,  Time: 0:03:08 
 90%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                | 1264/1407 [00:13<00:01, 110.97it/s]Iter:   8300,  Train Loss:  0.47,  Train Acc: 91.41%,  Val Loss:  0.29,  Val Acc: 91.07%,  Time: 0:03:09 
 97%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊     | 1360/1407 [00:14<00:00, 110.35it/s]Iter:   8400,  Train Loss:  0.49,  Train Acc: 84.38%,  Val Loss:  0.29,  Val Acc: 91.38%,  Time: 0:03:10 *
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1407/1407 [00:14<00:00, 94.18it/s]
Epoch [7/50]
  4%|█████▋                                                                                                                                                         | 50/1407 [00:00<00:11, 117.12it/s]Iter:   8500,  Train Loss:  0.54,  Train Acc: 89.06%,  Val Loss:  0.28,  Val Acc: 91.33%,  Time: 0:03:12 *
 11%|█████████████████▋                                                                                                                                            | 158/1407 [00:01<00:11, 112.10it/s]Iter:   8600,  Train Loss:  0.45,  Train Acc: 94.53%,  Val Loss:  0.29,  Val Acc: 91.14%,  Time: 0:03:13 
 18%|████████████████████████████▌                                                                                                                                 | 254/1407 [00:02<00:10, 110.60it/s]Iter:   8700,  Train Loss:  0.41,  Train Acc: 96.88%,  Val Loss:  0.29,  Val Acc: 90.96%,  Time: 0:03:14 
 25%|███████████████████████████████████████▎                                                                                                                      | 350/1407 [00:03<00:09, 111.03it/s]Iter:   8800,  Train Loss:  0.41,  Train Acc: 93.75%,  Val Loss:  0.28,  Val Acc: 91.22%,  Time: 0:03:15 
 33%|███████████████████████████████████████████████████▍                                                                                                          | 458/1407 [00:04<00:08, 111.41it/s]Iter:   8900,  Train Loss:  0.38,  Train Acc: 96.09%,  Val Loss:  0.28,  Val Acc: 91.10%,  Time: 0:03:16 
 39%|██████████████████████████████████████████████████████████████▏                                                                                               | 554/1407 [00:05<00:07, 110.16it/s]Iter:   9000,  Train Loss:  0.48,  Train Acc: 92.97%,  Val Loss:  0.29,  Val Acc: 91.21%,  Time: 0:03:17 
 46%|████████████████████████████████████████████████████████████████████████▉                                                                                     | 650/1407 [00:06<00:06, 110.27it/s]Iter:   9100,  Train Loss:  0.42,  Train Acc: 96.88%,  Val Loss:  0.28,  Val Acc: 91.07%,  Time: 0:03:18 
 54%|█████████████████████████████████████████████████████████████████████████████████████                                                                         | 758/1407 [00:07<00:05, 111.28it/s]Iter:   9200,  Train Loss:  0.42,  Train Acc: 95.31%,  Val Loss:  0.29,  Val Acc: 91.17%,  Time: 0:03:19 
 61%|███████████████████████████████████████████████████████████████████████████████████████████████▉                                                              | 854/1407 [00:08<00:05, 110.30it/s]Iter:   9300,  Train Loss:  0.46,  Train Acc: 92.19%,  Val Loss:  0.29,  Val Acc: 91.23%,  Time: 0:03:20 
 68%|██████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                                   | 950/1407 [00:09<00:04, 109.99it/s]Iter:   9400,  Train Loss:  0.53,  Train Acc: 92.97%,  Val Loss:  0.29,  Val Acc: 91.31%,  Time: 0:03:21 
 75%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                       | 1058/1407 [00:10<00:03, 111.45it/s]Iter:   9500,  Train Loss:  0.45,  Train Acc: 97.66%,  Val Loss:  0.29,  Val Acc: 91.03%,  Time: 0:03:22 
No optimization for a long time, auto-stopping...
 75%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                       | 1058/1407 [00:10<00:03, 96.40it/s]
```

预测结果如下：

```css
Test Loss:  0.27, Test Acc: 91.65%
Precision, Recall and F1-Score...
               precision    recall  f1-score   support

      finance     0.9398    0.8750    0.9063      1000
       realty     0.9382    0.9260    0.9321      1000
       stocks     0.8666    0.8770    0.8718      1000
    education     0.9579    0.9560    0.9570      1000
      science     0.8861    0.8790    0.8825      1000
      society     0.8554    0.9290    0.8907      1000
     politics     0.9186    0.8910    0.9046      1000
       sports     0.9441    0.9630    0.9535      1000
         game     0.9583    0.9190    0.9382      1000
entertainment     0.9091    0.9500    0.9291      1000

     accuracy                         0.9165     10000
    macro avg     0.9174    0.9165    0.9166     10000
 weighted avg     0.9174    0.9165    0.9166     10000

Confusion Matrix...
[[875  16  65   1   7  18   7   6   1   4]
 [  8 926  17   3   1  21   6   6   3   9]
 [ 35  18 877   1  28   6  29   3   0   3]
 [  0   2   2 956   4  16   4   3   1  12]
 [  1   3  20   8 879  26  16   7  22  18]
 [  3  13   1  13  13 929  13   2   3  10]
 [  9   5  21   9  16  37 891   3   2   7]
 [  0   2   3   0   3  10   3 963   1  15]
 [  0   0   4   4  37   6   1  12 919  17]
 [  0   2   2   3   4  17   0  15   7 950]]
Time usage: 0:00:00
```

