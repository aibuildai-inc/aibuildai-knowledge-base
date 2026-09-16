# 44th place solution

Competition: feedback-prize-english-language-learning
Rank: #44
Source: https://www.kaggle.com/c/feedback-prize-english-language-learning/discussion/371467

## Overview
Our solution is an ensemble of ridge regression and support vector regression, trained on predictions of 8 fine-tuned deberta models and 2 classical ml algorithms using 10-fold cross-validation and optimized with *Optuna* (see Fig. 1).[Figure 1: Our solution scheme].png?generation=1670625664430131&alt=media)
The scores are listed in the table below:
| **sample** | **score** |
| --- | --- |
| cross-validation | 0.44355 |
| public test | 0.43845 |
| private test | 0.435586 |

## [Our code](https://www.kaggle.com/code/andreyustinov/inference-notebook-stacking?scriptVersionId=112415397)

## What worked:
- Very careful cross-validating on 10 folds and own coefficients for each target field
- Mix of debertas
- Adding some classical ml (even with features based on deberta)
- Automatic weight tune
## What didn't work:
- Handcrafted and semi-automatic features (although some of them were not so bad)
- Sentence embeddings (especially for a `vocabulary` mark)
## Important Citations:
- [Ten debertas notebook](https://www.kaggle.com/code/jingwora1/fb3-deberta-family-inference-weight-tune)
- [Optuna](https://optuna.org)
## Thanks and Acknowledgements:
- My teammate Andrey Ustinov
- All those great and honorable persons who publish high-score notebooks (even if it blows the leaderboard up)
## Team members:
- [Andrey Ustinov](https://www.kaggle.com/andreyustinov) 
- [Innokentiy Humonen](https://www.kaggle.com/user303188)
