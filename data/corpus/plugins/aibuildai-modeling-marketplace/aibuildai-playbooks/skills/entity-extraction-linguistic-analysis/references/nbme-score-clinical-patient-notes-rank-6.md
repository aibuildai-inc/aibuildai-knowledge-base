# 6th place solution

Competition: nbme-score-clinical-patient-notes
Rank: #6
Source: https://www.kaggle.com/c/nbme-score-clinical-patient-notes/discussion/323237

It was nice to see the shake up fall kindly on us :) Congrats all and thank you NBME and Kaggle for hosting such an amazing competition.

### TL;DR
Our solution is based on knowledge distillation of an ensemble of DeBERTa models and efficient domain-finetuning utilizing pseudo-labeling of a big text-corpus of unlabeled data. We used a 2-level approach: In a first step we trained models on the given training data and used an ensemble of them to pseudo label the given unlabeled patient notes. In a second step we trained 4 models on the training + pseudo-labeled data. 

### Pre/Data processing
During the competition we tried a variety of data processing steps. While most of them did not help significantly we used them in different models just for the sake of creating diversity. For example we used

- special tokens for line breaks and medical acronyms; 
- lowercasing all caps text
- Mixup of labels 
- Token dropout

What did help significantly was MLM pre-training of weights, but unfortunately the benefit was redundant with pseudo-labeling which we performed at a later stage. Of course we also tried a lot of other things which did not help. We evaluated all models of the same 5 fold split and saw good agreement of CV score and public LB score (and private LB score, too). 

### Level1 Models
We used 

- deberta-large
- deberta-v2-xlarge
- deberta-v2-xxlarge
- deberta-v3-large

Backbones from huggingface repository and trained a number of different models on different preprocessing, seen above. We then generated pseudos for each set and saved the probabilities. Based on CV, we built the best blend of models, and created a pseudo set using a blend of the character level probabilities. CV score of this blend was around 0.897.

### Level2 Models
In a second stage we trained new DeBERTa models on training data + pseudos, namely

- deberta-large
- deberta-v2-xlarge
- deberta-v3-large

As you can see we did not have deberta-v2-xxlarge in our level2 models because it was too heavy for the final inference kernel. It still contributed significantly via the pseudo labeling and the related knowledge distillation. The final sub was 4 models, 3 folds each. 

Two things are quite interesting to note:
We experimented with adding more architectures to our level1 blend and tried a variety of backbones including BioBert, BioMegatron, RoBERTa, BART. But they did not improve the cv of our blend because they were too far behind the DeBERTa in terms of single model score
Given they are trained on (roughly) the same pseudo-blend the level2 models had little diversity. However a single model of those already achieves a Top10 position on LB

### Post processing
Not much postprocessing except stripping spaces and removing any predictions on line break characters. Both of these did help. 
Also, after Feedback we made sure to try WBF, xgboost and rnn stackers, all which did not help here 😁
