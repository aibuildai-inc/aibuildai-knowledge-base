# 19th place solution

Competition: llm-detect-ai-generated-text
Rank: #19
Source: https://www.kaggle.com/c/llm-detect-ai-generated-text/discussion/470181

Thanks to Kaggle, the host and the community for this competition!

I'm glad I survived the storm. A bit thirsty as I'm so close to gold, but given I had no clue how to score a medal up until a week or so ago, I feel it turned out fine for me. 

My main goal coming into the competition was moving up the ranks on datasets, notebooks and discussions. I feel I did really well there, thanks for all the upvotes! I might turn into discussions GM soon :) 

# Aproach
I approached this competition in 3 phases: 
1. Data generation. I expected the key is generating data similar to how the host generated it, and finding good CV/LB correlation. I made good progress there but never found CV/LB correlation. 
2. Implementing and testing single models and methods. I only started doing this seriously in January, and given lack of good CV and limited attempts at public LB, I feel I didn't fully reach the potential of these approaches (see a list below)
3. Ensembling. I started ensembling in the last week. The goal was to create a diverse and robust ensemble that will survive the shake. 
4. Postprocessing. Thanks to the sharing from @piotrkoz, I used his code to fix the train and test dataset for some of the models in ensemble on the last day. 

I didn't have time or energy to probe public LB much, which I think was good in hindsight. 

# Solution
Final solution is a weighted blend of:
1. Ridge classifier on V2 + sample of V4 datasets, public TF-IDF and test tokenization pipeline, spelling correction. 
2. Multinomial, SGD, Catboost models from the top public notebooks (V2, TF-IDF, test tokenization) with spelling correction
3. LightGBM with weighted sampling from V3 dataset based on source and a 2-step pipeline based on TF-IDF and test tokenization. First I train a lightweight model, take top 10k most important features, then train a heavier LGBM model on those selected features. 
4. Ranking based on `flesch_kincaid_grade_level` and `flesch_reading_ease metrics`.
5. Deberta-v3-large trained on a subset of V4 dataset. 
6. Mistral-instruct-7B trained with QLORA on V4 dataset, using @hotchpotch pipeline

# Failed experiments
Some experiments that didnt' work for me (reading other top solutions I probably didn't spend enought time to make them work):
1. Mistral-based reward model trained on pairs of original and generated text (new dataset I created in the last 2 days and didn't get to share)
2. Ghostbuster re-implementation with Llama probabilities. Since this performed so well for #1 team, I should probably revisit my code. 
3. Various models trained on quantized unigram and trigram probabilities of words in essays. 
4. Deberta-based ranking. I trained a model in multiple choice architecture, compared random pairs of essays in test, and then trained a Bradley-Terry model on those comparisons. 
5. NER model. I mixed random samples of human- and AI-generated essays and trained a NER model on tokens. It worked really well in my experience, I then used various features to predict text-based score. The best feature was number of consecutive tokens above certain probability threshold to make it AI generated. This is the interpretable method I shared in one of the discussions, and while it didn't work well on the LB, I feel it could be very useful. 
6. Ridge forest - I experimented with combining some ideas from RF/GBDT - feature and data subsampling - with linear classifiers like Ridge. It added a bit of robustness but also execution time, so I removed it at the end.

Thanks to everyone that shared ideas, datasets, code and memes (very much needed!) in the competition!
