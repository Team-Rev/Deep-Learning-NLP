import csv
import re
import shlex

import pandas as pd
import os
import load_data as data
from nltk.tokenize import word_tokenize
import numpy as np

# ppt 키워드 <-> 챕터 상호 연결
chapter_set = pd.read_csv('csv/result_ex.csv', delimiter='|')
title = []
root_dir = os.listdir('CCNA')

na_value = "('null',0.0)"
chapter_set = chapter_set.fillna(na_value)

for i in range(len(root_dir)):
    sub_dir = os.listdir('CCNA/'+f'{root_dir[i]}')
    for j in range(len(sub_dir)):
        title.append(sub_dir[j][0:-5])
chapter_title = pd.Series(title, name='text')
# total_set = pd.concat([chapter_set, chapter_title], axis=1) # csv 형식 chapter

keywords_set = chapter_set.values.tolist()

# 덤프 exam 로드 및 정제
data = data.load_data()
exams = data.get_data() # csv 형식 exam

text_exams = exams['linked_text'][:5].tolist()


# temp = list(chapter_set.columns)

# temp = chapter_set.loc[0] # 인덱스로 행 가져옴
# temp2 = temp.apply(lambda x: ' '.join([w for w in x.split()])) # 컬럼별로 잘라옴
# print(temp2[0]) # config 한개
# temp3 = str(temp[0]).split(',') # 잘라온거 콤마별로 잘라옴
# keyword = re.sub('[^a-z]','',temp3[0]) # 키워드 정규식으로 정제
# weight = re.sub('[^\d+.+\d]','',temp[1]) # 가중치 정규식으로 정제
# print(keyword)
# print(weight)
total_weight = 0
keywords_weight = []
for i, exam in enumerate(text_exams):
    tokens = word_tokenize(exam)
    keywords_weight.append([])
    for k in range(len(chapter_set)):
        for j, token in enumerate(tokens):
            total_weight = 0
            row = chapter_set.loc[k]
            for p in range(len(row)):
                row_list = row.apply(lambda x: ' '.join([w for w in x.split()]))
                key_weight = str(row_list[p]).split(',')
                keyword = re.sub('[^a-z]','',key_weight[0])
                weight = re.sub('[^\d+.+\d]','',key_weight[1])
                if token == keyword:
                    total_weight += float(weight)
        keywords_weight[i].append(total_weight)

w = open('weight.csv', 'w', newline='')
wr = csv.writer(w)

for index in keywords_weight:
    wr.writerow(index)





