# 6th place solution

Competition: jigsaw-multilingual-toxic-comment-classification
Rank: #6
Source: https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion/161095

Thanks to Jigsaw to hold such amazing competition! And also thanks to my teammate @tonyxu for diligence and brilliant ideas. This is our second jigsaw competition and also the first time to join such multilingual nlp competition. 

## Brief summary 
Before team merger, I mainly focused on how to train a single model with decent performance while my teammate worked on training diverse models, especially training on single language data and combine them together. Before merger, we both achieved 0.9487 on public leaderboard. As you can see, we worked in relatively different directions and achieved 0.9499 right after team merger. After team merger, we focused on training more diverse models for diversity, which gave us great boost but also caused us to ignore the post processing trick most top winners found.

## Detailed summary

### Data sampling
We used both english training data and its translation shared publicly. Since there were tremendous amount of training data available, the main issue was not too little but too much. I mainly adopted hard sampling technique to sample the training data. 

First, I included all positives and randomly sampled negatives to train a XLM-Roberta Large model and then used it to make predictions on all training data including all english data and translated jigsaw toxic comment. 

Then for english data, we set a threshold for negatives while we took all positives due to the data unbalance. The data with gap between the predicted probability and label greater than the threshold were all taken and the data with gap smaller than that were sampled. The same logic for translated data except the threshold setting. A single threshold may cause issues because translation led to information loss. So I set a value range, say [0.3, 0.7] and any data in this range were all taken. I didn't convert soft label to hard 0/1 label because it worked better in 
the last jigsaw competition. The number of training data is ~934k. XLM-Roberta Large trained on this data could reach 0.9469 on public lb and 0.9458 on private lb. 

### Multilingual Models
Our multilingual models mainly consist of `XLM-Roberta-Large`, `XLM-Roberta-base`, `Bert-multilingual-base-uncased`. My teammate also did language model finetuning on `jigsaw-unintended-bias` data. We don't know the exact number for each of them used in the final ensemble but the best public scores are 0.9469, 0.9364 and 0.9325 respectively. 

In terms of model structure, I used a vanilla classification head on the CLS token of the second to last layer or the concatenation of last four layers while my teammate took the [architecture of winning model](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/103280) in the last jigsaw competition.

### Monlingual Models
Just one week before the deadline, we found the monolingual language models in [huggingface communities](https://huggingface.co/models). We trained the models listed below on corresponding translated language, which gave us boost from 0.9507 to 0.9512 on public lb. 

&gt; * es:
        - https://huggingface.co/dccuchile/bert-base-spanish-wwm-uncased#
        - https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased#
        - https://github.com/dccuchile/beto        
* fr:
        - https://huggingface.co/camembert/camembert-large
        - https://huggingface.co/camembert-base
        - https://huggingface.co/flaubert/flaubert-large-cased
        - https://huggingface.co/flaubert/flaubert-base-uncased
        - https://huggingface.co/flaubert/flaubert-base-cased
* tr:
        - (128k vocabulary) https://huggingface.co/dbmdz/bert-base-turkish-128k-uncased
        - (128k vocabulary) https://huggingface.co/dbmdz/bert-base-turkish-128k-cased#
        - (32k vocabulary) https://huggingface.co/dbmdz/bert-base-turkish-uncased#
        - (32k vocabulary) https://huggingface.co/dbmdz/bert-base-turkish-cased#     
* it
        - https://huggingface.co/dbmdz/bert-base-italian-xxl-uncased#
        - https://huggingface.co/dbmdz/bert-base-italian-xxl-cased#
        - https://huggingface.co/dbmdz/bert-base-italian-uncased#
        - https://huggingface.co/dbmdz/bert-base-italian-cased#     
* pt
        - https://huggingface.co/neuralmind/bert-large-portuguese-cased#
        - https://huggingface.co/neuralmind/bert-base-portuguese-cased#      
* ru
        - https://huggingface.co/DeepPavlov/bert-base-bg-cs-pl-ru-cased
        - https://huggingface.co/DeepPavlov/rubert-base-cased#

### English models
To add more diversities, We also trained `roberta-large, bert-large, albert` on english training data and made prediction on data translated into English and to our surprise, this did improve the performance! And we also had all cross-lingual models predict on english-translated test data and added into our final ensemble.

### Training 
* [dynamic learning rate decay](https://arxiv.org/abs/1905.05583)
* Adam optimizer
* binary cross entropy loss function
* learning rate of 1e-6 to 3e-6 for 2 epochs for large models and 3 epochs for base models
* after training on english and translated data, we both finetuned further on validation dataset for 1-2 epochs.

My teammate had two different training schemes:
* took `jigsaw-toxic-comment-train.csv` and its translated datasets to train 7 models, one for each language and combine prediction together in the way below

```
test = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv")
sub = pd.read_csv('/kaggle/input/toxu-submissions/xlmroberta-large-lm-all/submission.csv')
test['toxic'] = 0.0
for lang in ['en', 'fr', 'es', 'it', 'pt', 'ru', 'tr']:
    lang_df = pd.read_csv(f'/kaggle/input/toxu-submissions/xlmroberta-large-lm-all/submission_{lang}.csv')
    idx = test[test['lang'] == lang].index
    test.loc[idx, 'toxic'] += lang_df.loc[idx, 'toxic'] * weight
    idx = test[test['lang'] != lang].index
    test.loc[idx, 'toxic'] += lang_df.loc[idx, 'toxic']
test['toxic'] /= 7
```
*  took `jigsaw-unintended-bias-train.csv` and its translation to train in the first epoch and used the same data in the above point

### Postprocessing
After seeing the trick used in nearly all the top solutions, we felt it was really a pity that we missed it because once my teammate forgot to average predictions for 7 language-specific models, causing some predictions to be greater and 0.0001 improvement. We didn't delve deeper into this due to the tiny improvement!

### Ensemble
Power averaging with power 3 in the [2nd place solution](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/100661) in the last jigsaw competition but didn't bring improvement compared with normal weighted average.

### What didn't work
* auxiliary task with 5 other labels in the training data
* sample weights for different languages
* extra labelled hatespeech data
* pesudo labelling
* multi-sample dropout
* cyclic learning rate
* checkpoint ensemble
* training multilingual bert from scratch on multilingual wikipedia data dump
