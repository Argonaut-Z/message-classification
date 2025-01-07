# 项目说明

## 信息自动投递项目背景介绍

今日头条是字节跳动旗下的核心产品之一，其推荐系统依赖于强大的自然语言处理（NLP）技术。短文本分类是该推荐系统的重要组成部分，用于自动将新闻内容按类别分类，为推荐引擎提供支持。

**目标**：
将新闻标题划分为固定的 10 个类别（如财经、娱乐、体育等），根据分类结果推荐更符合用户兴趣的内容，提升用户体验和平台收益。

**意义**：

1. **提高推荐精准度**：分类结果用于匹配用户兴趣，提升点击率和停留时长。
2. **增强数据管理效率**：自动分发新闻至对应频道，优化内容组织。
3. **创造商业价值**：增加订阅量、广告收益和用户粘性。

该项目是 NLP 技术在推荐系统中的典型应用，是字节跳动生态的重要技术支撑。

- 基于`Randomforest`的基线模型
- 基于`Fasttext`的基线模型
- 基于`BERT`的文本分类模型

## 1. 环境准备

在`conda`环境下创建虚拟环境并安装如下必要库：

```shell
conda create -n bert_classification python=3.10 -y
conda activate bert_classification

pip install torch==2.3.1 -i https://pypi.tuna.tsinghua.edu.cn/simple/ 
pip install numpy==1.23.5 -i https://pypi.tuna.tsinghua.edu.cn/simple/ 
pip install pandas==2.0.3 -i https://pypi.tuna.tsinghua.edu.cn/simple/ 
pip install scikit-learn==1.5.1 -i https://pypi.tuna.tsinghua.edu.cn/simple/ 
pip install jieba==0.42.1 -i https://pypi.tuna.tsinghua.edu.cn/simple/  
pip install ipykernel -i https://pypi.tuna.tsinghua.edu.cn/simple/ 
```

或者根据`requirements.txt`安装必要库：

```shell
pip install -r requirements.txt
```

windows系统下安装`fasttext`包，在本地已经下载好对应文件，执行如下命令：

```shell
pip install fasttext_wheel-0.9.2-cp310-cp310-win_amd64.whl
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
