import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from tensorflow.keras.preprocessing.text import Tokenizer
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import multiprocessing
import sys

vocab = Counter()

pd.set_option('display.max_colwidth', None)
s = WordNetLemmatizer()
p = PorterStemmer()
# l = LancasterStemmer()
index_count = []

# 불용어 리스트
custom_stop_words = ['computer', 'network', 'used', 'administrator', 'use', 'command', 'address', 'would', 'one', 'two','three', 'company', 'using']
stop_words = stopwords.words('english')
stop_words.extend(custom_stop_words)

# 샘플 데이터 로드
sample_data = pd.read_csv('link.csv', delimiter='|')


exams = sample_data[['exam','choice']]


# 텍스트 전처리
exams['exam'] = exams['exam'].str.replace("[^a-zA-Z]", " " ,regex=True)
exams['exam'] = exams['exam'].apply(lambda x: x.lower())
exams['exam'] = exams['exam'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 2 and w not in stop_words]))

exams['choice'] = exams['choice'].fillna(" ")
exams['choice'] = exams['choice'].str.replace("[^a-zA-Z]", " ", regex=True)
exams['choice'] = exams['choice'].apply(lambda x: x.lower())
exams['choice'] = exams['choice'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 2 and w not in stop_words]))

exams['linked_text'] = exams[['exam', 'choice']].apply(':'.join, axis=1)
# exams['linked_text'] = exams['linked_text'].apply(lambda x:' '.join([s.lemmatize(w) for w in x.split()]))
exams['linked_text'] = exams['linked_text'].apply(lambda x:' '.join([p.stem(w) for w in x.split()]))


#DoC2Vec 학습
# tagged_data = [TaggedDocument(words=word_tokenize(_d), tags=[str(i)]) for i, _d in enumerate(exams['linked_text'])]
#
# max_epochs = 1000
# model = Doc2Vec(vector_size=20,
#                 alpha=0.025,
#                 min_alpha=0.00025,
#                 min_count=1,
#                 dm=1)
# model.build_vocab(tagged_data)
#
# for epochs in range(max_epochs):
#     model.train(tagged_data,
#                 total_examples=model.corpus_count,
#                 epochs=model.epochs)
#     model.alpha -= 0.0002
#     model.min_alpha = model.alpha
#     print(epochs)

# model.save('d2v.model')

model = Doc2Vec.load('d2v.model')

# print(model.wv.most_similar('ospf', topn=3))

# 토큰화
tokenized_exam = [word_tokenize(exam) for exam in exams['linked_text']]

#TF-IDF
# tf = TfidfVectorizer()
# tf.fit(exams['linked_text'])
# tfidf = tf.transform(exams['linked_text']).toarray()
# word2id = defaultdict(lambda  : 0)
# for idx, feature in enumerate(tf.get_feature_names()):
#     word2id[feature] = idx
# for i, sent in enumerate(exams['linked_text']):
#     print('==== 문제[%d] ====' % i)
#     print([(token, tfidf[i, word2id[token]]) for token in sent.split()])



# Word2Vec 학습
model = Word2Vec(sentences=tokenized_exam, vector_size=100, window=3, min_count=10, workers=6, sample=0.001, sg=1)


model_result = model.wv.most_similar('switch')
print(model_result)
# model.wv.save_word2vec_format('ccna_w2v')

