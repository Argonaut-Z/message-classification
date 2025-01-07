import time
import jieba
import fasttext


# 服务框架使用Flask，导入工具包
from flask import Flask
from flask import request
app = Flask(__name__)

# 导入发生http请求的requests工具
import requests

# 加载自定义的停用词字典
jieba.load_userdict('../data/stopwords.txt')

# 提供已经训练好的模型路径+名字
model_save_path = 'toutiao_fasttext_20241227_111904.bin'

# 实例化fasttext对象，并加载模型参数用于推断，提供服务请求
model = fasttext.load_model(model_save_path)
print('FastText模型实例化完毕...')

# 设定投满分服务的路由和请求方法
@app.route('/v1/main_server/', methods=["POST"])
def main_server():
    """
    处理 POST 请求的主要服务方法
    """
    uid = request.form.get('uid', '')
    text = request.form.get('text', '')

    print(f"收到的请求参数: uid={uid}, text={text}")

    if not text:
        return "Error: text is empty", 400

    input_text = ' '.join(jieba.lcut(text))
    print(f"分词后的文本: {input_text}")

    res = model.predict(input_text)
    if not res or not res[0]:
        return "Error: prediction failed", 500

    predict_name = res[0][0]
    print(f"预测结果: {res}, 返回类别: {predict_name}")

    return predict_name
