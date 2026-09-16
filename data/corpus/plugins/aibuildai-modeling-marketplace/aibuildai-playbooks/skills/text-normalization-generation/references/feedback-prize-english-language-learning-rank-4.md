# 4th place solutioin

Competition: feedback-prize-english-language-learning
Rank: #4
Source: https://www.kaggle.com/c/feedback-prize-english-language-learning/discussion/369621

## Overview

My solution is stacking a total of 22 models. All the models trained in this competition are the same folds (4fold).



## What Worked 

- [High Impact] Train SVR / Ridge using pre-trained model embeddings
  - Extracted features from the last 4 layers of the 38 pre-trained models and used forward feature selection to explorate the best SVR
  - Trained ridge model with optimal embeddings combination for SVR
  - SVR is my best single model (CV: 0.4467)
- [High Impact] Pseudo Labeling
  - Two patterns are used for each models
      - Pre-train with pseudo labels and fine-tune with only the given train data afterwards
      - Concatenate pseudo labels with the given train data, and train all this data
  - The amount of FB1 data used as pseudo label
      - use only FB1 data similar to FB3
      - use all FB1 data (but only few model)
  - Repeat over and over again
- [High Impact] Ridge and LGB stacking
  - CV: 0.4425(Ridge), 0.4443(LGB), 0.4423(Weighted Average of Ridge and LGB)
  - Train Ridge using the predictions of fine-tuned models as input values
  - Train LGB using the predictions and meta-features created by [readability](https://pypi.org/project/readability/)
- [Middle Impact] Add special token ('\n')
  - CV increase (0.001)
- [Low Impact] Shorten sequence length
  - Almost use 512 length
  - Inference time can be reduced
  - For model diversity, train few models with 1500 length
- [Low Impact] Tips for stable training
  - Full precision training
  - Layer wise learning rate decay
  - Cosine learning rate scheduler
  - Hyperparameter tuning

## What Didn’t Worked 

- AWP, FGM
- Train SVR using fine-tuned model embeddings
- Last Layer Re-initialization
- Regular MLM
- Predict punctuation errors and their statistics use as meta-feature in the stacking model
  - https://www.kaggle.com/code/nulldata/deep-learning-powered-punctuation-corrector/notebook
- Predict the quality rating of discourse element (FB2 task targets) and their statistics use as meta-feature in the stacking model
  - CV score and public score increase, but private score decrease (0.0002)
- Fine-tune spelling correction model as convention-specific model
  - https://huggingface.co/oliverguhr/spelling-correction-english-base?text=lets+do+a+comparsion
- Fine-tune pre-trained weights from top solution in the past competition
  - https://www.kaggle.com/competitions/feedback-prize-2021/discussion/313424

## Important Citations: 

- [Petfinder.my - Pawpularity Contest 1st place solution](https://www.kaggle.com/c/petfinder-pawpularity-score/discussion/300938)
- [CommonLiit Readability Prize 2nd place solution](https://www.kaggle.com/c/commonlitreadabilityprize/discussion/258328)
- [Feedback Prize - Predicting Effective Arguments 1st place solution](https://www.kaggle.com/competitions/feedback-prize-effectiveness/discussion/347536)
- [RAPIDS SVR - CV 0.450 - LB 0.44x](https://www.kaggle.com/code/cdeotte/rapids-svr-cv-0-450-lb-0-44x)
- [FB3 / Deberta-v3-base baseline [train]](https://www.kaggle.com/code/yasufuminakama/fb3-deberta-v3-base-baseline-train)

## Thanks and Acknowledgements: 

Thanks to the organizers and Kaggle for a very exciting competition!
I will use what I learned here to do my best in the next competition.

## Team Members:

- @shuheigoda
