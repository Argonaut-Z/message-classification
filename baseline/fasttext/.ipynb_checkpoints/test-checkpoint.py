import requests
import time

# 定义请求 URL 和请求数据
url = "http://0.0.0.0:5001/v1/main_server/"

data = {"uid": "AI-6-202104", "text": "公共英语(PETS)写作中常见的逻辑词汇汇总"}
data = {"uid": "AI-6-202104", "text": "Grubby参赛 ASSEMBLY今晚开战!"}

start_time = time.time()

# 向服务发送 POST 请求
res = requests.post(url, data=data)

cost_time = time.time() - start_time

# 打印返回结果
print("输入文本：", data['text'])
print("分类结果：", res.text)
print("单条样本预测耗时：{:.2f} ms".format(cost_time * 1000))