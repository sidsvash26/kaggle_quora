#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 01:42:09 2017

@author: sidvash
"""

import pandas as pd
import numpy as np
import pickle

data_path = '../../kaggle_quora_data/'

#Get the k-core features from public script:
# https://www.kaggle.com/gsmafra/string-based-k-core-magic-features/code

question_kcores = pd.read_csv(data_path + 'question_kcores.csv')

k_core_dict = question_kcores.set_index('question')['kcores'].to_dict()

df_train = pd.read_csv('../../kaggle_quora_data/train.csv')
df_train['question1'] = df_train['question1'].map(lambda x: str(x).lower())
df_train['question2'] = df_train['question2'].map(lambda x: str(x).lower())

def extract_q1kcore(row):
    try:
        return k_core_dict[row['question1']]
    except:
        return 1

def extract_q2kcore(row):
    try:
        return k_core_dict[row['question2']]
    except:
        return 1

def max_core(row):
    return max(row['q1_kcore'], row['q2_kcore'])

df_train['q1_kcore'] = df_train.apply(extract_q1kcore, axis=1)
df_train['q2_kcore'] = df_train.apply(extract_q2kcore, axis=1)

df_train['max_kcore'] = df_train.apply(max_core, axis=1)

train_X = np.array(df_train[['q1_kcore', 'q2_kcore', 'max_kcore']])
pickle.dump(train_X, open('../../kaggle_quora_data/feats8_kcore_v1.sav', 'wb'))

