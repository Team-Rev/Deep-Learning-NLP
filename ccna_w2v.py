import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from gensim.models import Word2Vec
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from collections import Counter

vocab = Counter()

pd.set_option('display.max_colwidth', None)
s = WordNetLemmatizer()
p = PorterStemmer()
# l = LancasterStemmer()
index_count = []

# 불용어 리스트
custom_stop_words = ['computer', 'network', 'used', 'administrator', 'use', 'command', 'address', 'would', 'one', 'two', 'three', 'company', 'using']
stop_words = stopwords.words('english')
# print(stop_words)
stop_words.extend(custom_stop_words)

# 샘플 데이터 로드
sample_data = pd.read_csv('link.csv', delimiter='|')


exams = sample_data[['exam','choice']]


# # 텍스트 전처리
exams['exam'] = exams['exam'].str.replace("[^a-zA-Z]", " " ,regex=True)
exams['exam'] = exams['exam'].apply(lambda x: x.lower())
exams['exam'] = exams['exam'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 2 and w not in stop_words]))

exams['choice'] = exams['choice'].fillna(" ")
exams['choice'] = exams['choice'].str.replace("[^a-zA-Z]", " ", regex=True)
exams['choice'] = exams['choice'].apply(lambda x: x.lower())
exams['choice'] = exams['choice'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 2 and w not in stop_words]))

exams['linked_text'] = exams[['exam', 'choice']].apply(' '.join, axis=1)
# exams['linked_text'] = exams['linked_text'].apply(lambda x:' '.join([s.lemmatize(w) for w in x.split()]))
# exams['linked_text'] = exams['linked_text'].apply(lambda x:' '.join([p.stem(w) for w in x.split()]))
# 토큰화
tokenized_exam = [word_tokenize(exam) for exam in exams['linked_text']]






# Word2Vec 학습
model = Word2Vec(sentences=tokenized_exam, vector_size=100, window=3, min_count=10, workers=6, sample=0.001, sg=1)


model_result = model.wv.most_similar('switch')
print(model_result)
model.wv.save_word2vec_format('ccna_w2v')

