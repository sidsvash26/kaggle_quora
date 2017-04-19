#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 09:52:26 2017

@author: sidvash
"""

import pickle
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
import xgboost as xgb
import numpy as np
import pandas as pd

train_X1 = pickle.load(open('../../kaggle_quora_data/feats1_tfidf_train.sav', 'rb'))
train_X2 = pickle.load(open('../../kaggle_quora_data/feats2_match_train.sav', 'rb'))
train_X3 = pickle.load(open('../../kaggle_quora_data/feats3_glove_train.sav', 'rb'))

#Concatenate all features
train_X = np.concatenate((train_X1, train_X2, train_X3), axis=1)


#Load training target variable
data = pd.read_csv('../../kaggle_quora_data/train.csv')
train_y = np.array(data.is_duplicate)

def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=0):
        params = {}
        params["objective"] = "binary:logistic"
        params['eval_metric'] = 'logloss'
        params["eta"] = 0.02
        params["subsample"] = 0.7
        params["min_child_weight"] = 1
        params["colsample_bytree"] = 0.7
        params["max_depth"] = 4
        params["silent"] = 1
        params["seed"] = seed_val
        num_rounds = 500 
        plst = list(params.items())
        xgtrain = xgb.DMatrix(train_X, label=train_y)

        if test_y is not None:
                xgtest = xgb.DMatrix(test_X, label=test_y)
                watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
                model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=100, verbose_eval=10)
        else:
                xgtest = xgb.DMatrix(test_X)
                model = xgb.train(plst, xgtrain, num_rounds)
                
        pred_test_y = model.predict(xgtest)

        loss = 1
        if test_y is not None:
                loss = log_loss(test_y, pred_test_y)
                return pred_test_y, loss, model
        else:
            return pred_test_y, loss, model
        
#Re-sampling the data
train_X_dup = train_X[train_y==1]
train_X_non_dup = train_X[train_y==0]

train_X = np.vstack([train_X_non_dup, train_X_dup, train_X_non_dup, train_X_non_dup])
train_y = np.array([0]*train_X_non_dup.shape[0] + [1]*train_X_dup.shape[0] + [0]*train_X_non_dup.shape[0] + [0]*train_X_non_dup.shape[0])
del train_X_dup
del train_X_non_dup
print("Mean target rate : ",train_y.mean())


kf = KFold(n_splits=5, shuffle=True, random_state=2016)
for dev_index, val_index in kf.split(range(train_X.shape[0])):
    dev_X, val_X = train_X[dev_index,:], train_X[val_index,:]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    preds, lloss, model = runXGB(dev_X, dev_y, val_X, val_y)
    break

pickle.dump(model, open('../../kaggle_quora_data/model2_feat123.sav', 'wb'))


#[490]   train-logloss:0.352547  test-logloss:0.355942 -- only these features result in this much score on training