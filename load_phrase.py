import pandas as pd
import numpy as np
import re

class load_phrase:
    phrases = pd.read_csv('phrases.csv', delimiter='\n')
    phrases.columns = ['word']
    phrases['word'] = phrases['word'].apply(lambda x: x.lower())
    phrases['word'] = phrases['word'].apply(lambda x: re.sub(r"^\s+", "", x))
    phrases = np.array(phrases['word']).tolist()
    phrases = list(dict.fromkeys(phrases))

    def __init__(self):
        print('get phrases')

    def get_stopwords(self):
        return self.phrases

