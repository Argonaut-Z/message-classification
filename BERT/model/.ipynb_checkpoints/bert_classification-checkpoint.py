import torch
import torch.nn as nn
import os
from transformers import BertModel, BertTokenizer, BertConfig

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        # self.bert = BertModel.from_pretrained("bert-base-chinese")
        self.bert = BertModel.from_pretrained(config.bert_path, config=config.bert_config)
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context, seq_len, mask = x

        # _, pooled = self.bert(context, attention_mask=mask)
        # out = self.fc(pooled)
        
        outputs = self.bert(context, attention_mask=mask)
        pooled = outputs.pooler_output
        # print(pooled.shape)
        out = self.fc(pooled)
        return out
    

if __name__ == "__main__":
    import sys
    sys.path.append('../')
    
    from config.config_bert import Config
    
    dataset = "toutiao"
    config = Config(dataset)
    model = Model(config)
    print(model)