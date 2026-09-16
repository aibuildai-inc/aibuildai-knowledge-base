# 59th Place Solution (And My First Silver Medal)

Competition: jigsaw-multilingual-toxic-comment-classification
Rank: #59
Source: https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion/160903

# Overview

My two best submissions were from a stacking pipeline. I've trained several models on Kaggle and Colab, which I later stacked their predictions using various stacking techniques (not just the usual mean, median, minmax) to achieve higher LB scores. 

## The Two Best Submissions

- Stacking &gt; 12 models (used mean of the predictions of 12 submission files, from 9 single models and 3 blends) **(Private LB: 0.9469, Public LB: 0.9485)**
- Blend of stacked predictions from above that gave 0.9485 on the public LB with `ExtraTreesClassifier` predictions trained on the validation data with https://www.kaggle.com/jazivxt/howling-with-wolf-on-l-genpresse/notebook **(Private LB: 0.9471, Public LB: 0.9485)**



# Models Used For Stacking

- XLM-RoBERTa
- XLM-RoBERTa Large
- XLM-RoBERTa Large + MLM training
- ExtraTreesClassifier

P.S. Might have missed out one or two, will update here should I find any missing.

# How Models Were Trained

- For the MLM model (third above), I trained it on Kaggle 
- For the XLM-RoBERTa models, I trained them on Google Colab TPU

# Notebooks With Models Referenced Above

A big thanks to all whom contributed to the following notebooks, you have taught me quite a lot! I'm still relatively new to NLP, and your sharing has made me more comfortable and interested in the domain! :)

- https://www.kaggle.com/riblidezso/train-from-mlm-finetuned-xlm-roberta-large 
- https://www.kaggle.com/jazivxt/howling-with-wolf-on-l-genpresse
- https://www.kaggle.com/hamditarek/ensemble
- https://www.kaggle.com/sai11fkaneko/data-leak
- https://www.kaggle.com/shonenkov/tpu-inference-super-fast-xlmroberta
- https://www.kaggle.com/shonenkov/tpu-training-super-fast-xlmroberta
- https://www.kaggle.com/aiaiooas/parcor-regularised-classification
- https://www.kaggle.com/shahules/fine-tune-xlm-kfold-cv-0-93-lb

# Notebooks Used For Submission

- https://www.kaggle.com/khoongweihao/ensemble-ii-the-dark-side-of-stacking **(Mine)**
- https://www.kaggle.com/jazivxt/howling-with-wolf-on-l-genpresse **(Public)**

# Ensemble Methods That Did Not Work

- Bi-weighted correlations stacking (something that stemmed from my personal research, which worked on other regression and classification problems and datasets, just not this one sadly. It actually worked at the start, but in the last month of the competition I could not find a suitable configuration due to the lack of submissions. See my notebook above for details)
- Power-weighted stacking/blend (intention was to penalize the classification probabilities, idea came from the winning solution of the previous Jigsaw competition. Sadly this did not work out too)
- Distance-weighted stacking (main idea is to give higher weights to predictions with lower distances such as correlation/euclidean distance. This did not work too)
