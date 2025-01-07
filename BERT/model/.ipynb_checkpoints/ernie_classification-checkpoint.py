import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, BertTokenizer

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        # 初始化 ERNIE 模型
        self.ernie = BertModel.from_pretrained(config.ernie_path, config=config.ernie_config)
        self.fc = nn.Linear(config.hidden_size, config.num_classes)  # 分类层

    def forward(self, x):
        input_ids, seq_len, attention_mask = x  # 输入为 input_ids 和 attention_mask

        # 使用 ERNIE 模型输出
        outputs = self.ernie(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # 取最后一层第一个 token 的隐藏状态（类似于 [CLS]）
        cls_representation = last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        out = self.fc(cls_representation)  # 分类层输出
        
        return out


if __name__ == "__main__":
    import sys
    sys.path.append('../')
    
    from config.config_ernie import Config  # 假设你有 ERNIE 的配置文件

    dataset = "toutiao"
    config = Config(dataset)
    model = Model(config)
    print(model)

