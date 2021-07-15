import csv
from collections import defaultdict
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import load_data as data
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
import tensorflow as tf

chapters_title = []
chapters = []
chapter_line = []
root_dir = os.listdir('ccna_token')

# data = data.load_data()
# exam = data.get_data()

for i, chapter in enumerate(root_dir):
    chapter_text = ''
    chapter_line = []
    chapters_title.append(chapter[0:-9])
    f = open(f'ccna_token/{str(chapter)}')
    while True:
        line = f.readline()
        line = line.strip()
        if not line: break
        chapter_line.append(line)
    # chapter_line.append(chapter[0:-9])
    chapter_text = ' '.join(chapter_line)
    chapters.append(chapter_text)

# 원 핫 인코딩
# t = Tokenizer()
# t.fit_on_texts(chapters)
# print(t.word_index)

# tokenized = []
# for i, chapter in enumerate(chapters):
#     tokenized.append(word_tokenize(chapter))

# 단어 토큰화 csv 추출
# f = open('dataset.csv', 'w', newline='')
# wr = csv.writer(f)
# for text in tokenized:
#     wr.writerow(text)
# f.close()


# 텍스트 - 챕터 csv 추출 (토큰화 x)
# dataset = pd.Series(chapters, name='text')
# title = pd.Series(chapters_title, name='chapter')
# dataset = pd.concat([dataset, title], axis=1)
#
# dataset.to_csv('dataset.csv', index=False)

dataset = pd.read_csv('dataset.csv')

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(dataset, 0.2)






#--TF-IDF--
# tf = TfidfVectorizer()
# tf.fit(chapters)
# tfidf = tf.transform(chapters).toarray()
# word2id = defaultdict(lambda :0)
# for idx, feature in enumerate(tf.get_feature_names()):
#     word2id[feature] = idx

# f = open(f'result.csv', 'w', newline='')
# wr = csv.writer(f)
#
# for i, sent in enumerate(chapters):
#     result = ([(token, tfidf[i, word2id[token]]) for token in sent.split()])
#     result.sort(key=lambda x:-x[1])
#     wr.writerow(result)
# ---------





