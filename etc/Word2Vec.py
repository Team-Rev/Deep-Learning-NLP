import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


data = pd.read_csv('Modules 1 – 3_question.csv', names=['id','title','main','sub'], delimiter='|')

title = data['title']

token = []

stopwords = set(stopwords.words('english'))


for i in range(len(title)):
    token.append(word_tokenize(title[i])) # 토큰화

result = []

for w in range(len(token)):
    result.append([])
    for n in range(len(token[w])):
        if token[w][n] not in stopwords:
            if len(str(token[w][n])) > 2:
                result[w].append(token[w][n]) # 불용어 제거

tokenizer = Tokenizer()
# tokenizer.fit_on_texts(result)

vocab_size = 5
tokenizer = Tokenizer(num_words=vocab_size + 1)
tokenizer.fit_on_texts(result)


encoded = tokenizer.texts_to_sequences(result)
padded = pad_sequences(encoded, padding='post')
print(tokenizer.word_index)
print('-------------------Top 5 word inclusions----------------------')
print(tokenizer.texts_to_sequences(result))
