#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 01:17:51 2017

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

#A loooot of cleaning needed in this
train = pd.read_pickle('../../kaggle_quora_data/train_preprocess.pkl')
test = pd.read_pickle('../../kaggle_quora_data/test_preprocess.pkl')


documents = pd.Series(train.question1.tolist() + train.question2.tolist() + test.question1.tolist() + test.question2.tolist())


#documents = pd.read_pickle('../../kaggle_quora_data/documents.pkl')

dictionary = corpora.Dictionary(line.lower().split() for line in documents)
# remove stop words and words that appear only once
stop_ids = [dictionary.token2id[stopword] for stopword in stops if stopword in dictionary.token2id]
once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq == 1]
dictionary.filter_tokens(stop_ids + once_ids)  # remove stop words and words that appear only once
dictionary.compactify()  # remove gaps in id sequence after words that were removed

print(dictionary)
dictionary.save('/tmp/basic_ex1.dict')
#print(dictionary.token2id)   #leads to memory overload - takes time
class MyCorpus(object):
    def __iter__(self):
        for line in documents:
             # assume there's one document per line, tokens separated by whitespace
            yield dictionary.doc2bow(line.lower().split())
            
        
corpus = MyCorpus()

#save corpus in a location
corpora.MmCorpus.serialize('/tmp/corpus.mm', corpus) #takes around 10 mins

#just checking one corpus example:
#for vector in corpus:
#    print(vector)
#    break
    # load one vector into memory at a time

tfidf = models.TfidfModel(corpus)     #initialize a model takes round 12mins

#Apply transformation to whole corpus
corpus_tfidf = tfidf[corpus]       #doesn't save to memory, runs in a second

#Check an example
#for doc in corpus_tfidf:
#    print(len(doc))
#    break

# initialize an LSI transformation - took around 7 hours to run, 4gb i3
lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=300) 
#started at 16.30:29
lsi.save('/tmp/lsi_model_v1.lsi')

#Uncomment below to load this later
#lsi = models.LsiModel.load('/tmp/model.lsi')

# create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi
corpus_lsi = lsi[corpus_tfidf] 


def feats_tfidf(row):
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


train_X = np.vstack(np.array(train.apply(lambda row: feats_tfidf(row), axis=1)))

pickle.dump(train_X, open('../../kaggle_quora_data/feats1_tfidf_train.sav', 'wb'))
