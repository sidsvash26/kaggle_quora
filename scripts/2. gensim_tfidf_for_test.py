#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 15:41:30 2017

@author: sidvash
"""
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from gensim import matutils
from gensim import models
from gensim import corpora
from six import iteritems
import pandas as pd
from nltk.corpus import stopwords
stops = set(stopwords.words("english"))
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
import numpy as np
from scipy.stats import skew, kurtosis
import pickle
import time
from datetime import datetime


test = pd.read_pickle('../../kaggle_quora_data/test_preprocess.pkl')

dictionary = corpora.Dictionary.load('/tmp/basic_ex1.dict')

lsi = models.LsiModel.load('/tmp/lsi_model_v1.lsi')

def feats_tfidf(row):
    curr_index = row.test_id
    global t0
    if time.time() - t0 > 10:
        t0 = time.time()
        print("Time: %s : %d rows processed" % (str(datetime.now()), curr_index))
    
    
    out_list = []
    que1 = str(row['question1'])
    que2 = str(row['question2'])
    
    #Calculate que1 lsa vector
    que1_vec = []
    que1_bow = dictionary.doc2bow(que1.lower().split())
    que1_lsi = lsi[que1_bow]
    for (index,value) in que1_lsi:
        que1_vec.append(value)
    
    #Calculate que2 lsa vector
    que2_vec = []
    que2_bow = dictionary.doc2bow(que2.lower().split())
    que2_lsi = lsi[que2_bow]
    for (index,value) in que2_lsi:
        que2_vec.append(value)
    
    #drop some dimensions if they don't match
    if len(que1_vec) != len(que2_vec):
        if len(que1_vec) > len(que2_vec):
            que1_vec = que1_vec[:len(que2_vec)]
            que2_vec = que2_vec
        else:
            que1_vec = que1_vec
            que2_vec = que2_vec[:len(que1_vec)]
        
    #Calculate distances between lsa vectors
    try:
        lsa_cosine = cosine(que1_vec, que2_vec)
    except:
        lsa_cosine = 1
    
    
    lsa_cityblock = cityblock(que1_vec, que2_vec)
    lsa_jaccard = jaccard(que1_vec, que2_vec)
    lsa_canberra = canberra(que1_vec, que2_vec)
    
    try:
        lsa_euclidean = euclidean(que1_vec, que2_vec)
    except:
        lsa_euclidean = np.nan
        
    lsa_minkowski = minkowski(que1_vec, que2_vec,3)
    lsa_braycurtis = braycurtis(que1_vec, que2_vec)
    
    lsa_q1_skew = skew(que1_vec)
    lsa_q1_kurtosis = kurtosis(que1_vec)
    
    lsa_q2_skew = skew(que2_vec)
    lsa_q2_kurtosis = kurtosis(que2_vec)
    
    
    out_list.extend([lsa_cosine,lsa_cityblock,lsa_jaccard,lsa_canberra,lsa_euclidean, \
                     lsa_minkowski,lsa_braycurtis,lsa_q1_skew,lsa_q1_kurtosis,lsa_q2_skew, lsa_q2_kurtosis])
    
    return out_list

t0 = time.time()
test_X = np.vstack(np.array(test.apply(lambda row: feats_tfidf(row), axis=1)))

pickle.dump(test_X, open('../../kaggle_quora_data/feats1_tfidf_test.sav', 'wb'))