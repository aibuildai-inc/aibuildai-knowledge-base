# 24th Short Solution

Competition: feedback-prize-effectiveness
Rank: #24
Source: https://www.kaggle.com/c/feedback-prize-effectiveness/discussion/347418

First of all, congratulate to all winners and thank you for the organizers. 
Also, thank you for all competitors especially [@abhishek](https://www.kaggle.com/code/abhishek/tez-for-feedback-v2-0) and [@kashiwaba](https://www.kaggle.com/code/kashiwaba/train-deberta-v3-large-with-optimization-approach) who publish the kernels that I refer the most.

#### Phase 1 Token prediction and span prediction

For the classification, I use token prediction and span prediction. In the essay text, there are multiple target texts. In my solution, the model is trained to predict the effectiveness using specific token or average of text span. Furthermore, I find that adding CLS and SEP token at the beginning of target text affects the CV scores. Using the deberta-v3-large and deberta-v2-xlarge, I obtained the following results.
| Model  | Approach  | CV |
| --- | --- | --- | 
| deberta-v3-large  | span prediction | 0.6182 | 
| deberta-v3-large  | CLS/SEP, token prediction| 0.6165 | 
| deberta-v3-large  | CLS/SEP, span prediction | 0.6181 | 
| deberta-v2-xlarge  | span prediction| 0.6290| 
| deberta-v2-xlarge  | span prediction, overfitting| 0.7193 | 
| deberta-v2-xlarge  | CLS/SEP, token prediction| 0.6308 | 
| deberta-v2-xlarge  | CLS/SEP, token prediction, overfitting | 0.7404 | 

Interestingly, deberta-v2-xlarge shows lower CV score, but it boosts the CV after the ensemble.  
After simple ensemble (manual weight tuning, CV 0.589, 0.583).

#### Phase 2 Bayesian optimization for ensemble weight and LGBM

To improve the score, I first ensemble the models where the weights are optimized to minimize OOF CV using Bayesian optimization (CV 0.579). 
Then, I use LGBM to improve the score further, I add effectiveness of prev/next text and location informations of the text as the additional information (CV 0.578, LB 0.575, Private LB 0.573).

#### Summary - Useful attempt
-	Inference effectiveness using whole essay text 
-	Ensemble span and token classification, use CLS/SEP token
-	Weight optimization and LGBM to deal with additional features

The solutions of other competitors are always astonishing and attractive. I regret that MLM pretraining using previous competition and pseudo labeling could improve my score. Appreciate other competitors for sharing great solutions. My solution is in https://www.kaggle.com/learnitanyway/24th-inference-deberta-ensemble.
