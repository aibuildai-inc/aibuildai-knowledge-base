# Toxic Solution and Review (2nd Place)

Competition: jigsaw-toxic-severity-rating
Rank: #2
Source: https://www.kaggle.com/c/jigsaw-toxic-severity-rating/discussion/308938

First of all, congrats to the winners and everyone who enjoyed the competition! ! Although this is my first time participating in a competition, I really like this competition on Kaggle, especially the competitive atomsphere in there.

# Overview
To be honest I never thought of finishing in 2nd place, which is largely due to the luck, but I will say ''Thank you!'' to the luck for seeing positive results for my persistence with **''Bert''** Method.I have to say that we could marvel at and believe the performance of "Bert" in NLP world all the time. The **Roberta **helps me attain the success, even though its' public score is lower than linear model, but I find that the pinpoint of this competition perhaps is that **''Public leaderboard perhaps mislead us to be overfitting'' **

#Retrospective
What I remember fresh is probably in the first ten days of January, when many people started to increase the public leaderboard ranking crazily, which made me feel that the overall trend was not normal, so I made two decisions: 
1. Running program Locally in order to avoid paying more attention to the public leaderboard scores.
2. I chose to part ways with a partner who was supposed to be teaming up. (The story here is that we communicated and did a lot of groundwork and exploration togethera at the beginning, but in those crazy period in January, I disagree with his obsession with improving public scoring and trust linear model instead of BERT, (Perharps the submission score of Roberta only 0.79416, which is even lower than 0.8 and not to say the linear model(0.893+)), but it would cause some overfitting risks I realized when I learn more from papers and my professor.  
Obviously, that's the reason why we can see this remarkable 'Shake-up' from Public leaderboard to Private Leaderboard. Fortunately, the Luck resonate my persistence in BERT method. 

#About Dataset
I used data from the last toxic classification competition. Since the goal of this competition is to predict the toxicity of comments, which is a regression task, I weighted averaged several features of the orginal data.

# About Data Cleaning
BERT uses raw data when pretrained, which means that Pre-trained BERT is based on an uncleaned corpus. Actually, I find that Data Cleaning will change distribution and lose information. 
Usually, data cleaning can improve the performance of linear models, but could harm the performance of BERT. If the weight decay or dropout value is too large, the model performance will degrade. When we go deep, we could find that Linear models need to be cleaned to reduce feature size and reduce overfitting. The capacity of BERT is large enough that it is not very necessary.
So in the end, I make a huge change of the code, especially simpliying the data cleaning procedure

Comparison：
Apply Data cleaning：  Private Score：0.80826，  Public Score：0.77609
Delete some Data cleaning： Private Score：0.81363，  Public Score：0.79416

#About Feature Engineering
1. For text preprocessing, only simple tokenization methods are used. I didn't select the features manually because the model can learn the importance of each token.
2.The BERT model acts like a feature transformer, converting text data into a vector representation of length 768.

# About Train Method
I used BERT(Bidirectional Encoder Representations from Transformers) for this competition. BERT is currently the State-of-the-art language model that makes use of Transformer to learn contextual relations between words (or sub-words) in a text. The model I used in this competition consists of two parts: the RoBERTa base, and a multi-layer perceptron built upon it for regression. That‘s is the essence of idea why I choose this method and insist it until the competition is ended.

#About Esemble Weight
I randomly sampled 5 folds of data, each fold contained a proportion of the data and trained a separate Model on each fold. Then average up the predictions from the 5 models.

#Optimization
I used **Adam optimizer **as it is computationally efficient and works well with large data sets and large parameters. Besides, I also used a learning rate schedular to decrease the learning rate gradually based on the epoch. This has the effect of quickly learning good weights early and fine tuning them later.

#Avoid Overfitting
Focusing on using** Weight Decay, Dropout Layers and Early stopping **to prevent the models from overfitting. All of these keep the relative good generalization of model.
To be more specific, For **Earlystoping**, the stopping criteria will be based on the validation loss in each epoch.

Thanks everyone，I am happy everyone help new bird , I learn a lot from this competition!
