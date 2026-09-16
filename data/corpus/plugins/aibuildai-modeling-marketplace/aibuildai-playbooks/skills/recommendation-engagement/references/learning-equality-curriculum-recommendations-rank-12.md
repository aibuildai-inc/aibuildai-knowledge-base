# 12th place solution

Competition: learning-equality-curriculum-recommendations
Rank: #12
Source: https://www.kaggle.com/c/learning-equality-curriculum-recommendations/discussion/394886

First of all, I would like to thank kaggle and the staff for hosting such an interesting competition.
I'm really happy that I achieved my goal of reaching the solo gold tier.

# 1. Summary 
 My solution consists of 3 stages. In the 1st stage, I created embeddings using Arcface and generated candidates. In the 2nd stage, I used a transformer-based rerank model with the distances from the first stage as features. In the 3rd stage, I used the 64-dimensional embeddings from the first stage obtained via SVD, distances, and predictions from the second stage to create rerank models using 1DCNN, LGBM, and MLP. I then performed rank ensemble, set a threshold, and selected final content ids. Finally, I added post-processing to fill in the gaps based on distances for cases where there were no content IDs.



# 2. 1st stage Arcface

## 2.0 Cross validation
I used stratified group k-fold to group the data up to grandparents into one group.

## 2.1 Feature engineering

My text creation was greatly boosted by @conjuring92 post [REF](https://www.kaggle.com/competitions/learning-equality-curriculum-recommendations/discussion/376873)

topics: Level + [sep] + title : description + [sep] + context + [sep] + children + [sep] + parent_description + [sep] + brother title
content : kind + [sep] + language + [sep] + title + [sep] + description + [sep] + text

## 2.2 MLM

I conducted MLM using the features described in section 2.1, and referred to this code [REF](https://www.kaggle.com/code/takamichitoda/disaster-tweets-mlm) for the implementation. Thank you.
MLM boosted my cv + 0.010 @ 1fold. But, it was necessary to verify that the model performed well on all folds (not execute).

## 2.3 Arcface architecture

I believe that using Arcface was the most distinctive feature of my solution.

The following is a dataset for training.




The following is an overview of the 1 iteration process.




First, I set the topic id and content id as outputs, and set the topic id as input. Then, I calculate the loss for each output (loss1, loss2).
Next, I set the content id as input, and calculate the loss for each output (loss3, loss4).
Finally, I took the average of the four losses, perform backpropagation, and train the model.

The number of epochs was approximately 30 epochs, the margin was 0.0001, and the value of s was adjusted depending on the model (around 10-15). 

## 2.4 Ensemble

I created nine models including xlm-roberta-large, xlm-roberta-base, and mdeberta-v3-base, and then concatenated the outputs of these models to perform an ensemble.
One of them, I used the pseudo labeling. I used the items with 'has content' equals False. While individually weak, they proved effective when combined in the ensemble.

In order to avoid out of memory, I must devide the topics and content... (Adjusting the bugs was very difficult.)

## 2.5 Using fulltrain

Using fulltrain was also one of my features. Initially, I used 4kfold, but I realized that fulltrain was extremely powerful. In the end, I did not use 4kfold to calculate distance and instead used the full train models trained with 3 different seeds (public LB + 0.007).

## 2.6 (just reference 1st stage + rule base submit result)

cv : 0.65289, public lb : 0.695, private lb : 0.732

# 3. 2nd stage transformer base rerank

I found comments about overfitting in the discussion, but I did not experience it. I used folds consistently with the 1st stage. Moreover, it was mentioned in some discussions that reranking using transformers works up to a certain point, but beyond that, it no longer works. I also experienced it. However, by adding the distance from the 1st stage as input, I was able to obtain more cv results.

## 3.1 Feature engineering
I set the input as follows:
Simirarity : str (int((1-distance)*1000)) + [sep] + topics title : topics description + [special original defined sep] + topics context + [sep] + content title : content description : content text

## 3.2 Ensemble
model1 : xlm-roberta-large  cv : 0.66418
model2 : sentence-transformers/paraphrase-xlm-r-multilingual-v1  cv : 0.66039

model1 * 0.7 + model2 * 0.3 = cv 0.6668  ,public lb : 0.70118, private lb : 0.73884

# 4. 3rd stage LGBM,1dcnn, MLP

In the 3rd stage, we used the SVD 64-dimensional embeddings generated in the 1st stage, as well as language, distance, and predictions generated in the 2nd stage, as features. The results for each are as follows.

LGBM : cv 0.6644
1dcnn : cv 0.66450
mlp : cv 0.663156

# 5. Rank ensemble for final submission and postprocess

I performed a mean ensemble of the results obtained in sections 3.2 and 4, ranked by their respective scores.
In the post-processing, for the topics in which there were no results above the threshold, I established a specific number per language and used a filling technique based on the distance in the 1st stage.

final cv : 0.668755, public lb : 0.7023, private lb : 0.74044 (12 th)

# 6. Not working for me

- Changes in margin and s for each epoch at Arcface
- AWP
- augmentation by mixup
- Catboost, XGboost, Tabnet
- Knowledge Distillation

# 7. Acknowledgments

I couldn't get this score on our own. I am grateful to those who shared their knowledge in the past, those who teamed up with me, and everyone else! I respect to you.

Special thanks to this competition (using the code, dataset, and strategy)
@conjuring92, @takamichitoda, @ragnar123, @yasufuminakama (fb3 notebook)
