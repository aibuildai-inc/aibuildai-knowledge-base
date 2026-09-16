# 2nd solution tyu part

Competition: petfinder-adoption-prediction
Rank: #2
Source: https://www.kaggle.com/c/petfinder-adoption-prediction/discussion/89042#latest-516326

First of all, thank you very much Petfinder.my and kaggle for hosting such a fantastic competition. And congrats winners and all kagglers :) And I would like to express my great thanks to teammates.

This post is mainly takuoko, ynktk, u++’s part.
Here are our teams other parts.

[2nd Place Solution Summary](https://www.kaggle.com/c/petfinder-adoption-prediction/discussion/88773#latest-512687)
[2nd Place Solution about k_features and LGBM2](https://www.kaggle.com/c/petfinder-adoption-prediction/discussion/88812#latest-512246)   <br>
[g_features + XGB (part of 2nd place solution)](https://www.kaggle.com/c/petfinder-adoption-prediction/discussion/88963#latest-513873)  

# Feature engineering
I made 4000～features. But because of 7200sec training time, I selected importance top 1000 features. Interesting thing is that CV improved with fewer features, but dropped LB. I think fewer features model tends not to overfit, but the result was opposite.

| n_feature | CV | LB |
|:-----------|------------:|:------------:|
| all | 0.460 | 0.472 | 
| 1000 | 0.462 | 0.470 |
| 150 | 0.470 | 0.460 | 


### Text
I used different preprocessing pipeline for BOW features or embedding features. 

+ BOW features from description
1. puncts. ex. what's?→what's ?
2. misspell. ex. what's→what is
3. stemming. ex. going→go
4. number. ex. 3cats→3 cats
5. BOW/tf-idf→SVD/NMF/BM25

on 5, I used 3 methods. 1st default, 2nd 1-3 gram, 3rd character one. This idea comes from my avito solution. We can get diverse features from its.

+ embedding features from description
1. puncts.
2. misspell
3. number
4. embedding with glove and fasttex. When loading embeddings, I used [Quora 3rd solution](https://www.kaggle.com/wowfattie/3rd-place).
5. averaging embedding vectors per sentence.

We checked percentage of embedded vocabulary ont by one. Finally, 93% words get embedding vectors.
I tested other embeddings like elmo, bert, etc... But no improvement. I also tried [scratch word2vec, fasttext](https://www.slideshare.net/JinZhan/kaggle-avito-demand-prediction-challenge-9th-place-solution-124500050) or [finetune fasttext like Quora 4th](https://www.kaggle.com/kfujikawa/word2vec-fine-tuning). These also had no gain. I think it is because of few training data.

+ BOW features from meta.
I used all json files but kaeru and gege used only 1st one. This caused some diversity.

1. concat meta text. sentiment +annots_desc +breed
2. same as BOW features from description. This part used only default one.

###  image features
I also used all image files but kaeru and gege used only 1st one.

+ Features from densenet121, InceptionResnetV2.
I made reference [this benchmark](https://github.com/cgnorthcutt/benchmarking-keras-pytorch) and test some pretrained features like resnet152 pytorch, xception keras, densenet161 pytorch, etc… But this 2 combination had no problem.


### Aggregation
+ simple aggregation

+ ratio aggregation
ex. Fee of data is 100. Fee of aggregation with Breed is 30. ratio = 100 / 30 = 3.3. This means the pet is expensive.

+ diff aggregation
ex. Fee of data is 100. Fee of aggregation with Breed is 30. diff = 100 - 30 = 30. This means the pet is expensive.

The following features are importance top 15.
+ diff var Sterilized groupby RescuerID State
+ diff var Age groupby MaturitySize
+ ratio sum Sterilized groupby State
+ ratio count Age groupby RescuerID
+ median Age groupby RescuerID
+ ratio sum Age groupby State
+ diff mean Age groupby MaturitySize

### simple aggregation for json files
I used all json files, all features(maybe😉).

### external data
+ [emoji data](https://www.kaggle.com/thomasseleck/emoji-sentiment-data)
+ State data.
+ Breed data.
+ Breed ranking data.

External data was dirty. Some breeds name and states name are different from petfinder data. Mainly u++ cleaned these data.
We also cleaned Breed1 Breed2. Some 

###  Matrix Factorization
+ BOW-&gt;LDA/SVD
I used all categorical feature combinations.

### interactions features
Age * Quantity
Age / Quantity
PhotoAmt / Quantity

# Feature selection
importance top 1000 features. There is some more imformation in [other part](https://www.kaggle.com/c/petfinder-adoption-prediction/discussion/88773#latest-512687)


# validation strategy
Group K fold.
We tested Group K fold, Stratified K fold, Group Stratified K fold.
We decided to use it because of following 4 reasons.
 
1. LB score Stratified K fold=Group K fold&gt;Group Stratified K fold
2. CV score Stratified K fold&gt;&gt;Group K fold&gt;Group Stratified K fold
3. run time Stratified K fold&gt;&gt;Group K fold=Group Stratified K fold
4. There are no RescuerID that is in train and test. I think Stratified K fold cause heavily overfitting for CV.

# model
LGBM - CV(Group K fold)=0.462 , LB=0.470~
NN - CV(Group K fold)=0.435 , LB=???
Actually, I had better CV scored NN got CV=0.455. But I dropped it because of diversity and use scratch trained embeddings for 0.455NN. This NN got CV=0.447 and ensemble score improved well. But strangely, Public LB and Private LB was worse.

### NN model
 ![class][1]

 ![class2][2]

When training, I used following params.
+ Cyclic LR
+ epoch=20
+ batch size=256
+ optimizer=adam
+ loss=RMSE
+ embedding=not trainable

CV=455 NN is different from above.
It has following things.
+ se module
+ skip connection important top100 features
+ RNN-&gt;self attention

RNN Block also referenced [Quora 3rd place solution](https://www.kaggle.com/wowfattie/3rd-place). I tested [1st place solution model](https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/80568#latest-510114) and other RNN architecture, but this one was best.

I also tried following, but no gain.
+ GRU or LSTM like [talking data bestfitting](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56262#latest-433060)
+ libfm
+ adabound
+ scratch embedding

# other things that didn’t work
+ gege adversarial validation trick
+ other external data like cute image features, etc…
+ nima
+ A La Carte Embedding
+ word2vec, doc2vec, SCDV
+ target encoding, WOE
+ preprocssing chinese text
+ text static features like lenght, etc…
+ image statistic features like width, whiteness, etc…
+ catboost, xgboost, etc...
+ classification models

[1]: https://imgur.com/9THWht9.png
[2]: https://imgur.com/1885bPT.png
