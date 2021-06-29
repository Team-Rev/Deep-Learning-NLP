from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from nltk import word_tokenize

import load_data as load
import pandas as pd

data = load.load_data()
exams = data.get_data()
print(exams)

# DoC2Vec 학습
tagged_data = [TaggedDocument(words=word_tokenize(_d), tags=[str(i)]) for i, _d in enumerate(exams['linked_text'])]

max_epochs = 1000
model = Doc2Vec(vector_size=20,
                alpha=0.025,
                min_alpha=0.00025,
                min_count=1,
                dm=1)
model.build_vocab(tagged_data)

for epochs in range(max_epochs):
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.epochs)
    model.alpha -= 0.0002
    model.min_alpha = model.alpha
    print(epochs)

model.save('d2v.model')

model = Doc2Vec.load('d2v.model')

print(model.wv.most_similar('ospf', topn=3))