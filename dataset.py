import csv
from collections import defaultdict

import numpy as np
import pandas as pd
import os

from nltk import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

chapters_title = []
chapters = []
chapter_line = []
root_dir = os.listdir('ccna_token')


for chapter in root_dir:
    chapter_text = ''
    chapter_line = []
    chapters_title.append(chapter)
    f = open(f'ccna_token/{str(chapter)}')
    while True:
        line = f.readline()
        line = line.strip()
        if not line: break
        chapter_line.append(line)
    chapter_text = ' '.join(chapter_line)
    chapters.append(chapter_text)

# 원 핫 인코딩
t = Tokenizer()
t.fit_on_texts(chapters)
print(t.word_index)

tokenized = []
for i, chapter in enumerate(chapters):
    tokenized.append(word_tokenize(chapter))



# f = open('dataset.csv', 'w', newline='')
# wr = csv.writer(f)

# for text in tokenized:
#     wr.writerow(text)
# f.close()




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





