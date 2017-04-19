#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 01:45:51 2017

@author: sidvash
"""
#####################################################################
#                             MODEL 1                               #
#####################################################################


import pickle
import numpy as np
import xgboost as xgb
import pandas as pd

#Load model:
xg_model = pickle.load(open('../../kaggle_quora_data/model1_feat1_2_v2.sav', 'rb')) 

#Load test data features

test_X1 = pickle.load(open('../../kaggle_quora_data/feats1_tfidf_test.sav', 'rb'))
test_X2 = pickle.load(open('../../kaggle_quora_data/feats2_match_test.sav', 'rb'))

test_X = np.concatenate((test_X1, test_X2), axis=1)

#Predictions using model
xgtest = xgb.DMatrix(test_X)
preds = xg_model.predict(xgtest)

#Load test ids
test_data = pd.read_csv('../../kaggle_quora_data/sample_submission.csv')
ids = test_data.test_id


out_df = pd.DataFrame({"test_id": ids, "is_duplicate":preds})

list_col = out_df.columns.tolist()
list_col = list_col[-1:] + list_col[:-1]

out_df = out_df[list_col]


out_df.to_csv("../submissions/model1_feat1_2_v2.csv", index=False)

#####################################################################
#                             MODEL 2                               #
#####################################################################


import pickle
import numpy as np
import xgboost as xgb
import pandas as pd

#Load model:
xg_model = pickle.load(open('../../kaggle_quora_data/model2_feat123.sav', 'rb')) 

#Load test data features

test_X1 = pickle.load(open('../../kaggle_quora_data/feats1_tfidf_test.sav', 'rb'))
test_X2 = pickle.load(open('../../kaggle_quora_data/feats2_match_test.sav', 'rb'))
test_X3 = pickle.load(open('../../kaggle_quora_data/feats3_glove_test.sav', 'rb'))

test_X = np.concatenate((test_X1, test_X2, test_X3), axis=1)

#Predictions using model
xgtest = xgb.DMatrix(test_X)
print('predicting values')
preds = xg_model.predict(xgtest)
print('predictions done!!')
#Load test ids
test_data = pd.read_csv('../../kaggle_quora_data/sample_submission.csv')
ids = test_data.test_id


out_df = pd.DataFrame({"test_id": ids, "is_duplicate":preds})

list_col = out_df.columns.tolist()
list_col = list_col[-1:] + list_col[:-1]

out_df = out_df[list_col]


out_df.to_csv("../submissions/model2_feat123.csv", index=False)