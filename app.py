from classiflier import fasttext_classifier
import time

# create classifier
classifier = fasttext_classifier(model_path="model/classifier_comment.model",
                                 train_data="data/rate_train.txt",
                                 autotuneValidationFile="data/rate_valid.txt")

# model test
print(classifier.test('data/rate_valid.txt'))
print(classifier.test_label('data/rate_valid.txt'))

with open('data/train.txt', "r") as f:
    for r in f.readlines():
        r = r.replace("\n", "")
        print(classifier.predict(r, k=5, threshold=0.3), r)
        time.sleep(1)