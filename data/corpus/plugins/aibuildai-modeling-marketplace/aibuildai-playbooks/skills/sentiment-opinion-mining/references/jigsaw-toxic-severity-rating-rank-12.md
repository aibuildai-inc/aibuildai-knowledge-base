# 12th Solution – Trust Your CV

Competition: jigsaw-toxic-severity-rating
Rank: #12
Source: https://www.kaggle.com/c/jigsaw-toxic-severity-rating/discussion/306325

#####First of all, thank you for the organizers and other participants. I’ve learned many things including the importance of CV.
#### Solution
My solution is summarized as
- Use ‘validation_data.csv’ for the CV estimation. (Do not use it for the training)
- Training dataset - ‘jigsaw-unintended-bias-training.csv’ and ‘jigsaw-toxic-comment-train.csv’ in ‘[Jigsaw Multilingual Toxic Comment Classification](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification)’ challenge
- Model – TF-IDF with/wo text cleaning, distilroberta-base, roberta-base, roberta-large
- Ensemble of all trained model and TF-IDF

#### CV
Among my solution, the most important thing is to believe the CV. So, I firstly introduce the discussion that encourages me to focus on CV not LB.
- [Carefull, LB is only 5%](https://www.kaggle.com/c/jigsaw-toxic-severity-rating/discussion/287033)
- [Classic CV vs LB Discussion](https://www.kaggle.com/c/jigsaw-toxic-severity-rating/discussion/287147)
- [Higher CV, lower LB](https://www.kaggle.com/c/jigsaw-toxic-severity-rating/discussion/289089)

I find that it is hard to obtain CV score over 0.70 (using validation_data.csv), so my goal is to obtain **ensemble of model that stably gives** CV 0.7.

#### Dataset
It seems most of the public kernel use ’jigsaw-toxic-comment-train.csv’ from ‘Toxic Comment Classification Challenge’. It seems that data is subset of the data in ‘[Jigsaw Multilingual Toxic Comment Classification](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification)’. so I use the data from Multilingual challenge.
In the toxic comment and unintended bias, there are columns including ‘toxic’, ‘severe_toxic’, etc. I just give weight 1 for all cols except ‘severe_toxic’ that has weight 2. (It seems there is no huge difference on CV) 

#### Model and Ensemble
**TF-IDF** – the code for TF-IDF is barely modified from the referenced code
- [🔥 Ridge + fastText + Improved RoBERTa 🔥](https://www.kaggle.com/yamqwe/ridge-fasttext-improved-roberta#Toxic-clean-data)
- [TFIDF+RIDGE](https://www.kaggle.com/vitaleey/tfidf-ridge)

**Bert-like model** - distilroberta-base, roberta-base, and roberta-large. Trained with fasthug. 
- [Jigsaw Training - ULMFIT w/FastAi](https://www.kaggle.com/alibaba19/jigsaw-training-ulmfit-w-fastai)

**Submission (ensemble)** – use exhaustive search and equal weights for the submissions. Both works but equal weights is better.
-	Sub 1 (CV 0.706, Public 0.77805, Private 0.81029) - exhaustive search to maximizes CV 
-	Sub 2 (CV 0.705, Public 0.77631, Private 0.81044) - Similar weights except roberta-large (Actually, in case of roberta-large, I trained few epochs due to the computational budget...)  CV 705, Public 0.77631 Private 0.81044

The inference kernel is updated [here](https://www.kaggle.com/learnitanyway/12th-solution-jrstc-tf-idf-and-fastai). Thank you again for all participants and the organizers and congratulations to all winners.
