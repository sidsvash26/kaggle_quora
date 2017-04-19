#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 10:08:13 2017

@author: sidvash
"""
import time
import pandas as pd
import numpy as np
import difflib
import nltk
from nltk.corpus import stopwords
eng_stopwords = set(stopwords.words('english'))
from fuzzywuzzy import fuzz
from datetime import datetime
import math
import pickle

def diff_ratios(st1, st2):
    seq = difflib.SequenceMatcher()
    seq.set_seqs(str(st1).lower(), str(st2).lower())
    return seq.ratio()

train = pd.read_csv('../../kaggle_quora_data/train.csv')

t0 = time.time()

def feats_match(row):
    
    curr_index = row.id
    global t0
    if time.time() - t0 > 10:
        t0 = time.time()
        print("Time: %s : %d rows processed" % (str(datetime.now()), curr_index))
        

    
    out_list = []
    que1 = str(row['question1'])
    que2 = str(row['question2'])
    
    #length variables
    z_len1 = len(que1)
    z_len2 = len(que2)
    z_word_len1 = len(que1.split())
    z_word_len2 = len(que2.split())
    
    out_list.extend([z_len1,z_len2, z_word_len1, z_word_len2])
    
    #Fuzzy match variables:
    fuzz_qratio = fuzz.QRatio(que1, que2)
    fuzz_WRatio = fuzz.WRatio(que1, que2)
    fuzz_partial_ratio = fuzz.partial_ratio(que1, que2)
    fuzz_partial_token_set_ratio = fuzz.partial_token_set_ratio(que1, que2)
    fuzz_partial_token_sort_ratio = fuzz.partial_token_sort_ratio(que1, que2)
    fuzz_token_set_ratio = fuzz.token_set_ratio(que1, que2)
    fuzz_token_sort_ratio = fuzz.token_sort_ratio(que1, que2)
    
    out_list.extend([fuzz_qratio,fuzz_WRatio,fuzz_partial_ratio,fuzz_partial_token_set_ratio, \
                    fuzz_partial_token_sort_ratio,fuzz_token_set_ratio,fuzz_token_sort_ratio]) 
    
    #Match variables
    q1_nouns = [w for w, t in nltk.pos_tag(nltk.word_tokenize(str(que1).lower())) if t[:1] in ['N']]
    q2_nouns = [w for w, t in nltk.pos_tag(nltk.word_tokenize(str(que2).lower())) if t[:1] in ['N']]
    unigrams_que1 = [word for word in que1.lower().split() if word not in eng_stopwords]
    unigrams_que2 = [word for word in que2.lower().split() if word not in eng_stopwords]
    
    z_noun_match = sum([1 for w in q1_nouns if w in q2_nouns])
    z_match_ratio = diff_ratios(que1, que2)
    common_unig_len = len(set(unigrams_que1).intersection(set(unigrams_que2)))
    common_unig_ratio = float(common_unig_len) / max(len(set(unigrams_que1).union(set(unigrams_que2))),1)
    
    out_list.extend([z_noun_match,z_match_ratio,common_unig_len,common_unig_ratio])
    
    # get bigram features #
    bigrams_que1 = [i for i in nltk.ngrams(unigrams_que1, 2)]
    bigrams_que2 = [i for i in nltk.ngrams(unigrams_que2, 2)]
    common_bigrams_len = len(set(bigrams_que1).intersection(set(bigrams_que2)))
    common_bigrams_ratio = float(common_bigrams_len) / max(len(set(bigrams_que1).union(set(bigrams_que2))),1)
    out_list.extend([common_bigrams_len, common_bigrams_ratio])

    # get trigram features #
    trigrams_que1 = [i for i in nltk.ngrams(unigrams_que1, 3)]
    trigrams_que2 = [i for i in nltk.ngrams(unigrams_que2, 3)]
    common_trigrams_len = len(set(trigrams_que1).intersection(set(trigrams_que2)))
    common_trigrams_ratio = float(common_trigrams_len) / max(len(set(trigrams_que1).union(set(trigrams_que2))),1)
    out_list.extend([common_trigrams_len, common_trigrams_ratio])
    
    return out_list

#Takes around 1 hour to run
train_X2 = np.vstack(np.array(train.apply(lambda row: feats_match(row), axis=1)))

pickle.dump(train_X2, open('../../kaggle_quora_data/feats2_match_train.sav', 'wb'))

    