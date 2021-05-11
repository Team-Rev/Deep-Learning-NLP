import pandas as pd
from nltk.corpus import stopwords
from gensim import corpora
import gensim
import pyLDAvis.gensim_models

data = pd.read_csv('Modules 1 – 3_question.csv', names=['id','title','main','sub'], delimiter='|')

titles = data['title']
title = pd.DataFrame({'title':titles})
# 텍스트 전처리
title['clean_tle'] = title['title'].str.replace("[^a-zA-Z0-9]"," ", regex=True)
title['clean_tle'] = title['clean_tle'].apply(lambda x : ' '.join([w for w in x.split() if len(w) > 3]))
title['clean_tle'] = title['clean_tle'].apply(lambda x : x.lower())


stop_words = stopwords.words('english')
tokenized_tle = title['clean_tle'].apply(lambda x:x.split()) # 토큰화
tokenized_tle = tokenized_tle.apply(lambda x: [item for item in x if item not in stop_words]) # 불용어 제거

dictionary = corpora.Dictionary(tokenized_tle)
corpus = [dictionary.doc2bow(text) for text in tokenized_tle]

NUM_TOPICS = 20
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=NUM_TOPICS, id2word=dictionary, passes=15)
topics = ldamodel.print_topics(num_words=3)

#
# vis = pyLDAvis.gensim_models.prepare(ldamodel, corpus, dictionary)
# pyLDAvis.save_html(vis, 'LDA_Visualization.html')