---
language: zh
pipeline_tag: fill-mask
widget:
- text: "今天[MASK]情很好"
---

# albert_chinese_base

This a albert_chinese_base model from [Google's github](https://github.com/google-research/ALBERT)  
converted by huggingface's [script](https://github.com/huggingface/transformers/blob/master/src/transformers/convert_albert_original_tf_checkpoint_to_pytorch.py)


## Notice
*Support AutoTokenizer*

Since sentencepiece is not used in albert_chinese_base model   
you have to call BertTokenizer instead of AlbertTokenizer !!!    
we can eval it using an example on MaskedLM   
   
由於 albert_chinese_base 模型沒有用 sentencepiece   
用AlbertTokenizer會載不進詞表，因此需要改用BertTokenizer !!!   
我們可以跑MaskedLM預測來驗證這個做法是否正確   
   
## Justify (驗證有效性)
```python
from transformers import AutoTokenizer, AlbertForMaskedLM
import torch
from torch.nn.functional import softmax

pretrained = 'voidful/albert_chinese_base'
tokenizer = AutoTokenizer.from_pretrained(pretrained)
model = AlbertForMaskedLM.from_pretrained(pretrained)

inputtext = "今天[MASK]情很好"

maskpos = tokenizer.encode(inputtext, add_special_tokens=True).index(103)

input_ids = torch.tensor(tokenizer.encode(inputtext, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
outputs = model(input_ids, labels=input_ids)
loss, prediction_scores = outputs[:2]
logit_prob = softmax(prediction_scores[0, maskpos],dim=-1).data.tolist()
predicted_index = torch.argmax(prediction_scores[0, maskpos]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
print(predicted_token, logit_prob[predicted_index])
```
Result: `感 0.36333346366882324`   
