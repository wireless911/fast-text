import os
import fasttext
from fasttext import load_model as _load_model
from fasttext.FastText import _FastText

"""
          训练一个监督模型, 返回一个模型对象

          @param input:           训练数据文件路径
          @param lr:              学习率
          @param dim:             向量维度
          @param ws:              cbow模型时使用
          @param epoch:           次数
          @param minCount:        词频阈值, 小于该值在初始化时会过滤掉
          @param minCountLabel:   类别阈值，类别小于该值初始化时会过滤掉
          @param minn:            构造subword时最小char个数
          @param maxn:            构造subword时最大char个数
          @param neg:             负采样
          @param wordNgrams:      n-gram个数
          @param loss:            损失函数类型, softmax, ns: 负采样, hs: 分层softmax
          @param bucket:          词扩充大小, [A, B]: A语料中包含的词向量, B不在语料中的词向量
          @param thread:          线程个数, 每个线程处理输入数据的一段, 0号线程负责loss输出
          @param lrUpdateRate:    学习率更新
          @param t:               负采样阈值
          @param label:           类别前缀
          @param verbose:         ??
          @param pretrainedVectors: 预训练的词向量文件路径, 如果word出现在文件夹中初始化不再随机
          @return model object
 """


def load_model(model_path: str) -> _FastText:
    """Load a model given a filepath and return a model object."""
    return _load_model(model_path)


def train_model(model_path: str,
                train_data: str,
                autotuneValidationFile: str = "") -> _FastText:
    """Train an supervised model and return a model object"""

    # autotune train args
    train_kwargs = {
        "input": train_data,
        "label": "__label__",
        "autotuneValidationFile": autotuneValidationFile
    }
    # train args
    if autotuneValidationFile is "":
        train_kwargs.update({
            "dim": 100,
            "epoch": 200,
            "lr": 0.5,
            "wordNgrams": 2,
            "loss": "softmax",
        })

    # train model
    classifier = fasttext.train_supervised(**train_kwargs)
    classifier.save_model(model_path)

    return classifier


def fasttext_classifier(model_path: str = None,
                        train_data: str = None,
                        autotuneValidationFile: str = ""):
    """This function should be used to create a classifier. It will call functions such as load_model or train_model."""

    if not any([model_path, train_data]):
        raise ValueError("Your parameters are not complete Your parameters are not complete,"
                         "make sure your following parameter {params} is passed in".format(
                          params="model_path,train_data"))

    if os.path.isfile(model_path):
        classifier = load_model(model_path)
    else:
        classifier = train_model(model_path, train_data, autotuneValidationFile)
    return classifier


if __name__ == '__main__':
    classifier = fasttext_classifier(model_path="model/classifier_comment.model", train_data="data/rate_train.txt",autotuneValidationFile="data/rate_valid.txt")
