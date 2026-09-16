# 3rd Place - Short Solution

Competition: feedback-prize-effectiveness
Rank: #3
Source: https://www.kaggle.com/c/feedback-prize-effectiveness/discussion/347371

First of all congratulations to my teammate @conjuring92 for becoming a Kaggle Competitions Master! Very well deserved!!

This a very brief writeup of our solution which was a team effort of team **Darjeeling Tea**: @harshit92 @conjuring92 and me. 

Detailed writeup coming soon by @conjuring92  EDIT: Added here: https://www.kaggle.com/competitions/feedback-prize-effectiveness/discussion/347433

**Things that worked:**
- Span classification approach with LSTM head and multi-head attention
- Span MLM 
- Data augmentation using T5
- AWP/Token Mask Augmentation/Multi-sample dropout
- Prompt based learning
- 2nd level meta models - LSTM / LGB

**Things that did not work:**
- Regular MLM
- Pseudo Labeling
- Mixout 

Our best single arch was 10fold deberta-large - Public LB **0.563** Private LB **0.566** (gold zone)
Our most efficient sub was deberta-large trained on all train data + aug data - Public LB **0.565**/ Private **0.569** in 10mins
