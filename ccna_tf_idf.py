from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer

import load_data as data

data = data.load_data()
exams = data.get_data()

#TF-IDF
tf = TfidfVectorizer()
tf.fit(exams['linked_text'])
tfidf = tf.transform(exams['linked_text']).toarray()
word2id = defaultdict(lambda  : 0)
for idx, feature in enumerate(tf.get_feature_names()):
    word2id[feature] = idx
for i, sent in enumerate(exams['linked_text']):
    print('==== 문제[%d] ====' % i)
    print([(token, tfidf[i, word2id[token]]) for token in sent.split()])