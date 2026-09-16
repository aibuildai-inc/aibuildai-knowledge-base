# 11th place solution(Hui Qin's)

Competition: jigsaw-multilingual-toxic-comment-classification
Rank: #11
Source: https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion/160961

Congratulations all winners.  Thanks Kaggle supply this funny competition.  Thanks my teammates @Yang Zhang @MaChaogong @Hikkiiiiiiiii @Morphy



It is my first to solve the Multi-Lingual problem. Thanks for those great starter kernels:
https://www.kaggle.com/miklgr500/jigsaw-tpu-bert-two-stage-training
https://www.kaggle.com/xhlulu/jigsaw-tpu-xlm-roberta
I follow these codes at the very beginning and quickly got a lb 0.9415 single xlmr large model. 

After working 3 months for this problem, I got something useful tricks. 

1) What is the best training policy ? 
There are three training policies : one stage training,two stage training and three stage training.
a) One stage training means that we always train the same  data  in training .  
This policy always get the lower scores.  But we can get some diversity.
<br>
b) Two stage training means that we will have two different data for training.  e.g.  
We firstly train on "Toxic Comment Classification" dataset +"Unintended Bias in Toxicity" dataset . 
After that we train on 8k validation data.     
This policy always get the higher scores. 
<br>
c) Three stage training means that we will have three different data for training.  e.g. 
We firstly train on "Toxic Comment Classification" dataset +"Unintended Bias in Toxicity" dataset . 
Secondly  we train on 200k open subtitle data.    
Finally we train on 8k validation data. 
This policy always get the similar scores with Two stage training. But we get the higher private scores.
For example, 
my xlmr large model got  lb 0.9407, private score 0.9414.
my another xlmr large model got  lb 0.9420, private score 0.9424. 

Therefore , Three stage training is the best training policy.  I don't know why. 
<br>

2) How to use data? 

At the very beginning , my  teammate found that when we train on "Toxic Comment Classification" dataset +"Unintended Bias in Toxicity" dataset ,the valid auc  are always lower than training on "Toxic Comment Classification" dataset only.   To get a higher valid auc, we can not combine many different data to train.   
Before  9 days ago, I found that if we use our best single model to predict the training data's target and use the predicting target as the label instead of the 0/1 label.Then we can get a higher valid auc score.  e.g.  We use the our xlmr large model(lb 0.9411) to predict "Toxic Comment Classification" dataset and get the predicted target of them.  Something like these :
id toxic preds
1     0    0.341
2     1     0.813
3     0    0.211

Then we use the predicted probabilities as label target to train.  Using this trick, we always get a 0.003 boost in valid auc score.  This trick can use on other datasets which has 0/1 labels.  After using this trick on both "Toxic Comment Classification" dataset  and "Unintended Bias in Toxicity" dataset , we can use them to train our model. The more data, the higher valid auc score. 

Based on it, I trained a xlmr large model with 790k training data("Toxic Comment Classification" dataset  and "Unintended Bias in Toxicity" dataset and open subtitle dataset) . 
it got  a lb 0.9464 score and a 0.9445 private score. 



That's all .  Hope these two tricks can help you.
