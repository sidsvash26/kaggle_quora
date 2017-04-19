#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 03:11:10 2017

@author: sidvash
"""

import os
import pickle
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
from nltk import word_tokenize
import re
from tqdm import tqdm

stop_words = set(stopwords.words("english"))
stop_words.update('?')

train_df = pd.read_csv('../../kaggle_quora_data/train.csv', encoding='ISO-8859-1')
#train_df.drop(['id', 'qid1', 'qid2'], axis=1, inplace=True)

test_df = pd.read_csv('../../kaggle_quora_data/test.csv', encoding='ISO-8859-1')
#Basic pre-processing in the text
def string_replace(s):
    s = str(s)
    s = s.lower()
    s=s.replace("?", " ")
    
    #Seperators        
    s = re.sub(r"([a-zA-Z])\.([a-zA-Z)])", r"\1 \2", s) #sep '.' b/w letters
    #s=re.sub(r"([0-9])([a-zA-Z])", r"\1 \2", s) #sep alpha anumeric
    #s=re.sub(r"([a-zA-Z])([0-9])", r"\1 \2", s)
    #s=re.sub(r"([a-z])([A-Z])", r"\1 \2",s) #sep lowercase and uppercase letter
    
    #substitute more than one consecutive dots to a space
    s = re.sub(r"([.][.]+)", r" ", s)
             
        #removes any " adjacent to an alphabet
    s = re.sub(r"\"([a-zA-Z])", r" \1", s) 
    s = re.sub(r"([a-zA-Z])\"", r"\1 ", s) 
        
        #removes ' from highlighted words/phrase
    s = re.sub(r"( )(')([ a-zA-Z 0-9]+)(')", r" \3 ", s) 
    s = re.sub(r"(')([ a-zA-Z 0-9]+)(')( )", r" \2 ", s) 
        
       #removes any , adjacent to an alphabet
    s = re.sub(r"\,([a-zA-Z])", r" \1", s) 
    s = re.sub(r"([a-zA-Z])\,", r"\1 ", s) 
        
    s = re.sub(r"([a-zA-Z])\/([a-zA-Z])", r"\1 \2", s)  #sep '/' b/w letters
    
    #sep '/' b/w letters and number
    s = re.sub(r"([0-9])\/([a-zA-Z])", r"\1 \2", s)
    s = re.sub(r"([a-zA-Z])\/([0-9])", r"\1 \2", s)
              
    s=re.sub(r"([a-zA-Z])(\.) ", r"\1 ", s) #removes dot after any alphabet
    s=re.sub(r"([0-9])\,([0-9])", r"\1\2", s) #removing commas in b/w numbers
    s=re.sub(r"[()]", r" ", s) #removes open and close brackets
    
    #replacements
    s=s.replace("-", " ")
    s=s.replace("*", " ")
    s=s.replace("#", " ")
    s=s.replace(";", " ")
    s=s.replace("$", " ")
    s=s.replace("%", " ")
    s=s.replace("?", " ")
    s=s.replace(":", " ")
    s=s.replace("[math]", " [math] ")
    s=s.replace("[/math]", " [/math] " )

    #last substitution to save space
    s=re.sub(r"[  ]+", r" ", s) #substitutes double space to single
    
    return s

train_df['question1'] = train_df['question1'].map(lambda x: string_replace(x))
train_df['question2'] = train_df['question2'].map(lambda x: string_replace(x))
print("Basic pre-processing done in training..")


test_df['question1'] = test_df['question1'].map(lambda x: string_replace(x))
test_df['question2'] = test_df['question2'].map(lambda x: string_replace(x))
print("Basic pre-processing done in test data..")

train_df.to_pickle('../../kaggle_quora_data/train_preprocess.pkl')
test_df.to_pickle('../../kaggle_quora_data/test_preprocess.pkl')

#Examples sentences:
#str1 = "How many all time views do you have on your Quora answers (not questions) stats?"
#str2 = "How do I become a football/soccer manager/coach? What are some steps I can take to start?"
#str3 = "What pathway should I take to become a football (soccer) manager?"
#str4 = """ How do you define "human being?" """
#str5 = "how is the word 'discomfit' complexion used in a sentence. What is Kaufmich.com? How can I open a WhatsApp database (crypt8) without using a rooted phone? "
#str6 = "why does [math]\mathbb{r}^4[/math] have infinitely: many differential structures "
#str7 = """ What is the easiest way to become a billionaire($)? How do I deactivate the app "App Lock"? """
#str8 = "What is [math]x[/math] if [math]x+\left(\dfrac{1}{x}\right) =0[/math]?"
"""
str9 = "What is your review of Love in Thoughts (2004 movie)? \
        Which 2G Band (900/1800Mhz) was sold: during 2G spectrum scam? \
        Who will win the Premier League 2015-16? \
        Who will win the 2015/2016 Barclays Premier League?"      
"""

str10 = "can we cancel a 'tatkal waiting list' ticket? \
        how do i recover deleted photos from a samsung galaxy \
        s6/s5/s4/note?  abdomen pain on deep breathing...what can be the \
        cause I'm also thinking about don't doin nothing but 1/2 cup \
        of water is meta-human and 1,200 numbrs are base/trble stuff \
        should men read '50 shades of grey'"

del str10
#df_check = data[data.is_duplicate==1]
#df_check = train_df[train_df.is_duplicate==1]