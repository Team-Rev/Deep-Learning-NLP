import pandas as pd
from nltk.corpus import stopwords


class load_data:
    sample_data = pd.read_csv('link.csv', delimiter='|')
    # 텍스트 전처리
    exams = sample_data[['exam', 'choice']]
    exams['exam'] = exams['exam'].str.replace("[^a-zA-Z]", " ", regex=True)
    exams['exam'] = exams['exam'].apply(lambda x: x.lower())
    exams['choice'] = exams['choice'].fillna(" ")
    exams['choice'] = exams['choice'].str.replace("[^a-zA-Z]", " ", regex=True)
    exams['choice'] = exams['choice'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 2]))
    exams['choice'] = exams['choice'].apply(lambda x: x.lower())
    exams['linked_text'] = exams[['exam', 'choice']].apply(' '.join, axis=1)

    # exams['linked_text'] = exams['linked_text'].apply(lambda x:' '.join([s.lemmatize(w) for w in x.split()]))
    # exams['linked_text'] = exams['linked_text'].apply(lambda x:' '.join([p.stem(w) for w in x.split()]))

    def __init__(self):
        print('get data')



    def get_data(self):
        return self.exams
