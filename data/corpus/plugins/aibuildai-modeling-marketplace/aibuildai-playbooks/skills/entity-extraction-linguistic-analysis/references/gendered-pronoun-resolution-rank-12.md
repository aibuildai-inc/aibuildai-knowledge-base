# 12 place writeup

Competition: gendered-pronoun-resolution
Rank: #12
Source: https://www.kaggle.com/c/gendered-pronoun-resolution/discussion/90417#latest-523378

Our approach was based on extracting co-attention scores from pretrained  bert model. 

So this are selected lines for our preprocessing pipeline:

```yaml
  preprocess: 
    - try_search_near:  #  try to search for a mention of the names that is located closer to pronoun in question
    - replace_unknowns_preprocessor: #replace names that are not the part or pipeline 
         gender: true #try to select name of appropriate gender when gender statistics is known
         cased: false # use lower case
         lastNames: true # try to replace last names to well known lastnames
    - bert_encode:   #extract attention scores from bert 
        maxToken: true  #if named entity spans for several tokens take token with maximum attention score
    - disk-cache:
        split: true  
    - extract_token_bert_scores_3class:  
        startLayer: 9  #start from 9 layer of birt (was selected based on cv)

```
Then after this pipeline we have applied two models:

1) NN:
```
  net:
    - flatten
    - dropout: 0.5
    - dense: [144, relu]
    - dense: [3, softmax]
```
2) LGBM - with hyperparameter search

in bose cases we have used startified 10 fold, CV and blended models accross folds.

**Big Error:** We forgot to use test augmenation in our final predictions(you may replace names to a different one that are known to BERT and average predictions) , this costed us gold medal.

Regards,
Pavel
