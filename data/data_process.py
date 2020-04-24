import pandas as pd
import jieba
import re
import collections


mapper_tag = {
    '产品': '__label__product',
    '活动': '__label__activity',
    '赠品': '__label__gift',
    '客服': '__label__artificial',
    '物流': '__label__transfer',
    '扣单延迟发货': '__label__ship',
    '宝贝描述': '__label__detail',
    '发票问题': '__label__invoice',
    '其他': '__label__others'

}




# 去除停用词
def remove_stop_words(seg_list):
    stopwords = [line.strip() for line in open("stop_words.txt", 'r', encoding='utf-8').readlines()]
    seg_list = [seg for seg in seg_list if seg not in stopwords]
    return seg_list


def data_preprocess():
    """
    数据预处理
    :return:
    """

    df_comment = pd.read_excel('rates.xlsx')

    train_data = []
    for comment in df_comment['Content']:
        try:
            pattern = re.compile(u'\d+|[a-zA-Z]+\t|\n|\.|-|:|；|\)|\(|\?|，|。|、|【|】| |"')  # 定义正则表达式匹配模式：只匹配汉字
            comment = re.sub(pattern, '', comment.lower())  # 将符合模式的字符去除
        except Exception as e:
            comment = ""
        # 切词处理
        seg_list = jieba.cut(comment, cut_all=False)  # 精确模式分词
        seg_list = remove_stop_words(seg_list)


        train_line = " ".join(seg_list)
        train_data.append(train_line)

    # 保存带训练预料
    train_data = list(set(train_data))
    with open('train.txt', 'w', encoding='utf-8') as f:
        for line in train_data:
            f.write(" "+ ", "+ line + '\n')



if __name__ == '__main__':
    data_preprocess()
