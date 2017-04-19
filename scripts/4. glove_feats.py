#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 03:24:28 2017

@author: sidvash
"""
import pickle 
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
eng_stopwords = set(stopwords.words('english'))
import time
from datetime import datetime

glove_model = pickle.load(open('../../kaggle_quora_data/glove6B300d.sav', 'rb'))
print('Glover vector uploaded')

train = pd.read_pickle('../../kaggle_quora_data/train_preprocess.pkl')

def sent2vec(list1):
    temp_list = [w for w in list1 if w.isalpha()]
    M = []
    for w in temp_list:
        try:
            M.append(glove_model[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    
    return v / np.sqrt((v**2).sum())  

def feats_glove(row):
    #print time every 10 seconds
    curr_index = row.id
    global t0
    if time.time() - t0 > 10:
        t0 = time.time()
        print("Time: %s : %d rows processed" % (str(datetime.now()), curr_index))
    
    
    
    que1 = str(row['question1'])
    que2 = str(row['question2'])
    out_list = []
    # get unigram features #
    unigrams_que1 = [word for word in que1.lower().split() if word not in eng_stopwords]
    unigrams_que2 = [word for word in que2.lower().split() if word not in eng_stopwords]
    
    shared_words_in_q1 = [w for w in unigrams_que1 if w in unigrams_que2]
    shared_words_in_q2 = [w for w in unigrams_que2 if w in unigrams_que1]
    
    if len(unigrams_que1) == 0 or len(unigrams_que2) == 0:
        common_word_ratio = 0
    else:
        common_word_ratio = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(unigrams_que1) + len(unigrams_que2))
        
    out_list.extend([common_word_ratio])
    
    
    #Glove features
    wmd = min(glove_model.wmdistance(unigrams_que1, unigrams_que2), 12)
    
    q1_vec = sent2vec(unigrams_que1)
    q2_vec = sent2vec(unigrams_que2)
    
    #vector distances
    try:
        glove_cosine = cosine(q1_vec, q2_vec)
        
    except:
        glove_cosine = 1
    
    glove_cityblock = cityblock(q1_vec, q2_vec)
    glove_jaccard = jaccard(q1_vec, q2_vec)
    glove_canberra = canberra(q1_vec, q2_vec)
    
    try:
        glove_euclidean = euclidean(q1_vec, q2_vec)
    except:
        glove_euclidean = np.nan

    glove_minkowski = minkowski(np.nan_to_num(q1_vec), np.nan_to_num(q2_vec),3)
    glove_braycurtis = braycurtis(q1_vec, q2_vec)
    
    glove_q1_skew = skew(q1_vec)
    glove_q1_kurtosis = kurtosis(q1_vec)
    
    glove_q2_skew = skew(q2_vec)
    glove_q2_kurtosis = kurtosis(q2_vec)
    
    

    out_list.extend([wmd,glove_cosine,glove_cityblock,glove_jaccard,glove_canberra, \
                     glove_euclidean,glove_minkowski,glove_braycurtis, \
                     glove_q1_skew,glove_q1_kurtosis,glove_q2_skew,glove_q2_kurtosis])


    return out_list

t0 = time.time()
train_X3 = np.vstack(np.array(train.apply(lambda row: feats_glove(row), axis=1)))

pickle.dump(train_X3, open('../../kaggle_quora_data/feats3_glove_train.sav', 'wb'))
