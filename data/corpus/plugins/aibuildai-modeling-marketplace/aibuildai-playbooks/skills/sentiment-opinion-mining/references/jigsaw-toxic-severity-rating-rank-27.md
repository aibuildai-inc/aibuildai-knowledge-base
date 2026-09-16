# 27th Solution - What Worked and What Didn't for Me...

Competition: jigsaw-toxic-severity-rating
Rank: #27
Source: https://www.kaggle.com/c/jigsaw-toxic-severity-rating/discussion/306221

First of all congratulations, all for your hard work!

**Overview**
It was a weird experience, CV LB gap was insane and really unstable so I stop looking at LB scores after my first couple submissions unless something extraordinary happens (like really high or low score), otherwise I just concentrated on my CV scores. Then the question appeared: "Which CV to trust?" So I started by choosing CV strategy first:

**CV Strategy**
I tried several but at the end I decided to use @its7171's CV strategy [here](https://www.kaggle.com/its7171/jigsaw-cv-strategy). It made sense to me and giving somewhat more stable results.

**Data**
* Started with validation data only before getting into other datasets.
* Some of these others were pretty popular with other competitors. (Like old competition data or Ruddit etc.) then decided to rule some of them out to avoid possible leaks.
* [Measuring Hate Speech](https://www.kaggle.com/c/jigsaw-toxic-severity-rating/discussion/302060) with some preprocessing. I created less and more toxic pairs using several permutations and keeping the closest scored ones in dataset and added samples from it to main pipeline without replacement.

**Loss & Metric**
* Even though other loss functions like MSE etc. gave me decent CV scores (0.70+, 0.71+) at the end I decided to go with MarginRankingLoss.
* Metric wise I used what described in competition evaluation page...

**Model**
* My final ensemble consist of transformer based models only. Which include,
    - roberta variants mostly (Including distilbert),
    - deberta v3 variants,
    - bert variants (mostly toxic, hate ones)
    - bart

* Without external data my average CV was around `0.71` with external data included my CV didn't change much but a little bit less than the original scores but I decided to keep high scoring external data models to increase diversity at final ensemble.
* Models trained based on MarginRankingLoss, I didn't train the actual transformer blocks much (low learning rate for the backbones), instead I trained a final custom attention on top of their outputs and output layer.
* Fine-tuned these models with frequent validation checks. Usually converged around 3/4'th of first epoch.

Final submission was weighted sum of the rankings from these models.

**Things Might Have Worked**
* Models with sentence transformer embeddings. I created high dimensional matrices by concatenating different base transformer model embeddings. (This one was my second final selection and made barely into silver zone, didn't have time to include more models, it was only roberta and roberta-large)
* Combining different approaches at final ensemble (linear, transformer, embedding  ones etc.)

**Things Didn't Worked** 
* Linear Models with TF-IDF
* Models with word embeddings (Glove, fasttext etc.)
* Linear models with transformer embeddings combined together TF-IDF
* Pretraining masked language models on toxic datasets before finetuning on actual task.
* Some transformer approaches like Electra (I was expecting better CV score for discriminators)

At the end some of the earlier ensembles made much better than my final selection but that's the luck part in competition :) As I said this competition was an interesting experience for me, learned bunch of stuff on the road...
