# 3rd Place Solution - 5 seeds ensemble transformer

Competition: ubiquant-market-prediction
Rank: #3
Source: https://www.kaggle.com/c/ubiquant-market-prediction/discussion/338561

First, many thanks to the Kaggle team and Ubiquant market team for hosting this competition, especially for all efforts fixing bug at the first update. And congrats to all winners! I am very lucky winning 3rd place in my first financial comp. 

**Models**
6 layers transformer, max_seq_length=3500 investments

**Loss**
Optimize PCCLoss directly

**Training Method**
10 epochs on training data and 3 epochs on supplemental data

**Feature Engineering**
original 300 features

**Augmentation**
random zero (feature level) + random mask(sequence level)

**Validation Strategy**
last k (k=100,200,300) validation

**Ensemble Strategy**
5 seeds ensemble

**Rank Journey**
900+(public lb)->failed->7->7->4->3

**What Didn't Work**
feature clipping
avg features group by time_id(Maybe I'm wrong😂)
feature selection by corr
sample selecton or sample weight
target normalization or target clipping
tried lgb, mlp, 1dcnn but transformer outperform these models. I am too lazy to ensemble.
