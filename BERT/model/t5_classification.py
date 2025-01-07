import torch
import torch.nn as nn
from transformers import T5Model, T5Tokenizer, T5Config

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        # 初始化 T5 模型
        self.t5 = T5Model.from_pretrained(config.t5_path, config=config.t5_config)
        self.fc = nn.Linear(config.hidden_size, config.num_classes)  # 分类层

    def forward(self, x):
        input_ids, seq_len, attention_mask = x  # 输入为 input_ids 和 attention_mask
        
        # 使用 T5 编码器部分输出
        encoder_outputs = self.t5.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = encoder_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # 取第一个位置（句首）的表示
        cls_representation = last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        out = self.fc(cls_representation)  # 分类层输出
        
        return out


if __name__ == "__main__":
    import sys
    sys.path.append('../')
    
    from config.config_t5 import Config  # 假设你有 T5 的配置文件

    dataset = "toutiao"
    config = Config(dataset)
    model = Model(config)
    print(model)

    