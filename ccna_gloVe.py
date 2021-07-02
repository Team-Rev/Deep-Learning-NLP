from nltk import word_tokenize
from glove import Corpus, Glove
import load_data as data

data = data.load_data()
exams = data.get_data()

# 토큰화
tokenized_exam = [word_tokenize(exam) for exam in exams['linked_text']]

corpus = Corpus()
corpus.fit(tokenized_exam, window=5)

glove = Glove(no_components=100, learning_rate=0.05)
glove.fit(corpus.matrix, epochs=1000, no_threads=4, verbose=True)
glove.add_dictionary(corpus.dictionary)

model_result1 = glove.most_similar('vlan')

print(model_result1)