import load_data as data
from nltk.tokenize import word_tokenize, sent_tokenize
from gensim.models import FastText
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.stem import LancasterStemmer

p = PorterStemmer()
w = WordNetLemmatizer()
l = LancasterStemmer()
data = data.load_data()

exams = data.get_data()
result = [word_tokenize(sentence) for sentence in exams['linked_text']]


model = FastText(result, vector_size=100, window=5, min_count=5, workers=4, sg=1)
print(model.wv.most_similar('router'))