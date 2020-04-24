import re
from typing import Text
import jieba
import pandas as pd


def clean_txt(sentence: Text = ""):
    """
    数据清洗
    :param sentence: 文本
    :return res: 文本清洗结果
    """
    pattern = re.compile(r'\d+|[a-zA-Z]+\t|\n|\.|-|:|；|\)|\(|\?|，|。|、|【|】| |"')
    res = re.sub(pattern, '', sentence.lower()) if  isinstance(sentence, str) else ""

    return res


def stop_words():
    """
    去除停用词
    :return:　停用词列表
    """
    with open('user_dict/stop_words.txt', 'r', encoding='utf-8') as sw:
        return [line.strip() for line in sw]


def seg(sentence: Text, sw=stop_words):
    """
    分词
    :param sentence: 句子
    :param sw: 停用词函数
    :return: seg
    """
    sentence = clean_txt(sentence)
    return ' '.join([i for i in jieba.cut(sentence, cut_all=False) if i.strip() and i not in sw()])


def xlsx_to_txt():
    """
    数据预处理
    :return:
    """
    df_comment = pd.read_excel('data/rates.xlsx')
    train_data = []
    for comment in df_comment['Content']:
        if not comment:
            continue
        # 切词

        train_data.append(seg(comment))
    # 保存训练文件
    with open("data/train.txt", "w", encoding="utf8") as f:
        for data in train_data:
            # f.write(" , " + data + '\n')
            f.write( data+ '\n'  )


if __name__ == '__main__':
    xlsx_to_txt()
