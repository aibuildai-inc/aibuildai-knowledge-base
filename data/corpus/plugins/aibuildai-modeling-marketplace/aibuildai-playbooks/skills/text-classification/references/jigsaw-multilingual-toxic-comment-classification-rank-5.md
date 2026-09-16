# 5th place solution

Competition: jigsaw-multilingual-toxic-comment-classification
Rank: #5
Source: https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion/161997

Congratulations to all winners!

I would like to thank Kaggle and Jigsaw team for hosting this interesting competition. I am also grateful to this great community. Without those great public kernels I do not know how and where to start this project. I feel honored to end up this competition with a solo gold medal. It is a fantastic learning experience. 

As the private leaderboard is finalized, I decide to summarize what I have learned during this process. My solution has a lot in common with other top solutions. The two helpful techniques/tricks are **ensemble of diverse models** and **post-processing**. At certain point I also thought of using monolingual models as discussed in the [1st place solution](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion/160862). But I failed because I wrongly used XLM-R.

# Ensemble of diverse models

## Diverse models

For my personal understanding, langugae models are diverse if the train sets, tokenization methods or model architectures are different. In fact, I did not develop any methods from scratch. Instead, I started from public kernels, and improve them by trial and error. Below is a list of kernels I took advantage of for ensembling in my final submission. 

- [XLM-R TPU on PyTorch using Pseudolabeled (PL) opensubtitles](https://www.kaggle.com/shonenkov/tpu-training-super-fast-xlmroberta) by  [Alex Shonenkov](https://www.kaggle.com/shonenkov)

- [The first functioning XLM-R model](https://www.kaggle.com/xhlulu/jigsaw-tpu-xlm-roberta) by [xhlulu](https://www.kaggle.com/xhlulu)

- [XLM-R with fine-tuned MLM](https://www.kaggle.com/riblidezso/train-from-mlm-finetuned-xlm-roberta-large) by [Dezso Ribli](https://www.kaggle.com/riblidezso)

- [Two-stage BERT using translated dataset](https://www.kaggle.com/miklgr500/jigsaw-tpu-bert-two-stage-training) by [Michael Kazachok](https://www.kaggle.com/miklgr500)

- [NB-SVM](https://www.kaggle.com/jhoward/nb-svm-strong-linear-baseline) by [Jeremy Howard](https://www.kaggle.com/jhoward)

The final submission is an ensemble of 11 models, including two NB-SVM models trained with [Google API translated multilingual train set](https://www.kaggle.com/miklgr500/jigsaw-train-multilingual-coments-google-api), two BERT models trained with English train, validation and test sets, one XLM-R model trained with downsampled unintended bias train set and original validation set, one XLM-R model trained with PL test set, one XLM-R model trained using multilingual train set, and one XLM-R model with multilingual train set and PL test set, two XLM-R models with fined-tuned MLM, and three XLM-R models trained using PL opensubtitles. The ensemble weights are found by intuition and probing the public leaderboard score. The weights are

```python
weights = {}
weights['bert_1'] = weights['bert_2'] = 0.4/2
weights['nbsvm_1'] = weights['nbsvm_2'] = 0.035/2                         
weights['xlm_r_en'] =0.7; weights['xlm_r_pl_en'] = 0.3                
weights['xlm_r_multilingual']= 0.35; weights['xlm_r_pl_multilingual'] = 0.2
weights['xlm_r_mlm_en'] = weights['xlm_r_mlm_en_2'] = 0.25/2             
weights['xlm_r_mlm_multilingual'] = weights['xlm_r_mlm_multilingual_2'] = 0.25/2
weights['xlm_r_opensubtitle'] = weights['xlm_r_opensubtitle_2'] = weights['xlm_r_opensubtitle_3'] = 0.27/3 
```

# Post-processing

As highlighted in many other top solutions, post-processing contributes a lot to AUC score, since what matters is not the accuracy of single prediction, but the rank/distribution of all predictions. As we need to predict six-language comments in the test set, it is likely that the model may over-predict toxicity for one language while under-predict toxicity for another language. Therefore, scaling the prediction for individual language might be helpful. I first tried to put a 1.3 scaling coefficient for **fr, es and ru**. It gave my a boost of around 0.002 Public LB score for my best ensemble model. I also tried this with two single model predictions and also saw a similar trend. It gave me confidence that this should work generally, at least for the models I used. So for as long as two weeks, I experimented on the scaling coefficients, in the end I found the following coefficients that work best for me.

```python
scaling = {
    'fr': 1.175, 
    'ru': 0.975,
    'es': 1.475, 
    'it': 0.88,  
    'pt': 0.8, 
    'tr': 1.0 
}
```

Another language-wise post-processing trick is to separate the prediction somehow, it leads to a tiny LB score increase.

```python
def post_process(d):
    # d: list of numbers representing a distribution
    d = np.array(d)
    q_low, q_high = np.quantile(d, q=0.55), np.quantile(d, q=0.95)
    mask_nontoxic, mask_toxic = d&lt;=q_low, d&gt;=q_high
    d[mask_nontoxic] = 0.9*d[mask_nontoxic]
    d[mask_toxic] = 1.1*d[mask_toxic]
    return d

langs = test.lang.unique()
for language in langs:
    mask = test.lang == language
    sub.loc[mask, 'toxic'] = post_process(sub.loc[mask, 'toxic'])
```

# Things that worked

- Ensemble
- Post-processing: language-wise scaling
- Add another dense layer before the final dense layer
- Train on more balanced data
- Train on PL test set

# Things that did not work

- Ensemble including logistic regression models trained and predicted on either English or multilingual data

- Monolingual models using XLM-R

- Add more dense layers

- Learning rate scheduler

- Preprocess the data

- other post-processing techniques
