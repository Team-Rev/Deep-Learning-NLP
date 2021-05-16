import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np

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

#TF-IDF 행렬 생성

#역토큰화
detokenized_tle = []
for i in range(len(title)):
    t = ' '.join(tokenized_tle[i])
    detokenized_tle.append(t)
title['clean_tle'] = detokenized_tle

vetorizer = TfidfVectorizer(stop_words='english',
                            max_features=1000,
                            max_df=0.5,
                            smooth_idf=True)
X = vetorizer.fit_transform(title['clean_tle'])

# Topic Modeling

svd_model = TruncatedSVD(n_components=3, algorithm='randomized', n_iter=100, random_state=122)
svd_model.fit(X)

terms = vetorizer.get_feature_names()

def get_topics(components, feature_names, n=5):
    for idx, topic in enumerate(components):
        print("Topic %d:" % (idx+1), [(feature_names[i], topic[i].round(5))
                                      for i in topic.argsort()[:-n -1:-1]])
get_topics(svd_model.components_, terms)




