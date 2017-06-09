My scripts for Kaggle's "Quora Question Pairs" competition 
Competition URL - https://www.kaggle.com/c/quora-question-pairs

How to run the scripts:
1. Run all feature-engineering scripts in this order:
   1. pre_proces.py to 7. k_core_feats_v1.py
   2. All_magic_feats.ipynb
   3. feats10_Location_names_questions.ipynb
   4. Get word2vec features from this public forum - https://www.kaggle.com/c/quora-question-pairs/discussion/31284
2. Run XGB model based on above created features - XGB_model_final.ipynb

3. Run LSTM model from - LSTM_with_magic_and_more_feats.ipynb

4. Use average of LSTM and XGB to get the final predictions

My final standing on the private leaderboard is - 150/3394 (top 5 %)

Got my first Silver medal in this competition! :)

