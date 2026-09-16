# 18th place with only Kaggle and free Colab

Competition: nbme-score-clinical-patient-notes
Rank: #18
Source: https://www.kaggle.com/c/nbme-score-clinical-patient-notes/discussion/322958

First, big thanks to NBME and Kaggle for organizing such an interesting competition.

I am writing this mainly to show it's possible to get good results without owning or renting expensive GPUs even in competitions like this. I did most experiments on free Colab (T4 gpus) with with subsample of data and smaller models, when something looked promising I tested the idea with bigger model and more data and eventually trained and tested one full fold on Kaggle.

Because I had limited resources for experiments I went through all discussions in NBME competition and read solutions of similar competitions to find out what worked well so I want to thank everybody for sharing their knowledge in discussions. It was my first NLP competition and without kaggle community I wouldn't be able to reach such a good results. 

What worked for me:
- Keeping line breaks - changing the tokenizer as described [here](https://www.kaggle.com/competitions/feedback-prize-2021/discussion/313330)
- Converting texts and features to lower case
- Adding extra tokens for line breaks: ["\r", "\n", "\t"])
- Adding extra tokens for some shortcuts: ["cc","hpi", "pmhx", "pmh", "rx", "fhx", "shx", "phi", "meds", "sh", "fh", "ros", "psh", "hx", "pshx", "fmhx"]
- MLM pretraining (1 epoch, 0.2 mask probability, ~70% accuracy)
- Pseudo labels
- Post processing to fix spaces from [Roberta strikes back](https://www.kaggle.com/code/theoviel/roberta-strikes-back)

My ensemble is a bit of mix of everything because I didn't have resources to retrain my models so I worked with what i had:
1. deberta-v3-large (public: 0.888, private: 0.890) - weight: 0.4
5 folds, tokenizer not removing line breaks, lower case, added tokens, MLM pretraining and finetuned on pseudo labels (20k samples)
2. deberta-large-mnli (public: 0.885, private: 0.886)- weight: 0.2
trained on 95% training data, tokenizer not removing line breaks, lower case, added tokens, MLM pretraining and finetuned on pseudo labels (20k samples)
3. deberta-v3-large (public: ???, private: ???)- weight: 0.4
2 out of 5 folds (ran out of GPU time), normal tokenizer, no extra tokens, MLM pretraining and finetuned on pseudo labels (20k samples)

Models 1 and 2 used pseudolabels that were averaged from all models (1,2,3), which boosted the CV (probably leak) and single model score but didn't help in ensemble as much as I expected. So for model 3 I used pseudo labels only from it's own predictions (average across folds), that boosted CV (leak), single model score and helped in ensemble more (I think).

Final prediction was weighted sum of character level predictions from the models with post processing to fix spaces.

Special thanks to @theoviel for [Roberta strikes back](https://www.kaggle.com/code/theoviel/roberta-strikes-back), @yasufuminakama for great [baseline](https://www.kaggle.com/code/yasufuminakama/nbme-deberta-base-baseline-train) and @hengck23 for his [experiments](https://www.kaggle.com/competitions/nbme-score-clinical-patient-notes/discussion/315707).
