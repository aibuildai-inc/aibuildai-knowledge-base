# 40th place solution 0.553 - 0.555, +1050 positions

Competition: data-science-bowl-2019
Rank: #40
Source: https://www.kaggle.com/c/data-science-bowl-2019/discussion/127408

### I guess that many of those who are reading it understand how difficult it was to be at 1000+ position all the competition doing the right things at the same time. 😏 

## Of course I did not do:
- Any adjustment of test set features (adjustment by ratio of mean values was used in almost any public kernel).
- Tuning of round thresholds for final predictions (when we convert continuous values to discreat labels) to fit the train distribution of classes (that was used in almost any public kernel).

**It saved me from overfitting to public test distribution which was of course biased** (only 14% of data, it was clearly seen from my own experiments and this discussion: https://www.kaggle.com/c/data-science-bowl-2019/discussion/122767 )
**But at the same time it holds me at 1000+ LB position...**

I solved regression problem. It was the right way because kappa measure penalize the different errors unequally. Besides some popular features available in public kernels I used features based on “misses” and “rounds” and found them helpful. Additional custom features based on counters were used. After feature selection only 505 features were remained.

## Training:
- Using labeled test samples for training.
- **Custom RandGroupKfold** was used because sklearn GroupKfold does not have "seed" parameter to provide randomness.
- Ensamble of 15 LGBM models (3 CV iterations with 5 folds each, "feature fraction" parameter was changing at different folds).
- **Adversarial validation** via selection of random samples with unique IDs was used (200 iterations). **Median score** for each model was calculated and saved (it was used as a weight at model voting stage).
- Round **thresholds optimization** via OptimizedRounder was implemented **for each model individualy** based on whole train dataset. So each of 15 models has its own set of round thresholds what was used for final prediction.
- **Prediction confidence values were calculated based on ratio between the distance to nearest round value and interval length between adjacent round values**. These confidences were used as weights at model voting stage.
- Model voting stage was done with weights based on **median fold score** and **confidence values.**


I provided full results repeatability using SortedList, SortedDict, SortedSet (from sortedcontainers package) for features with "set" and "dict" as well as "seed" values for all models. It helped me a lot during model selection and submission stages.

## Things that did not help:
- Training dataset augmentation.
- Using kappa measures as evaluation approach for stopping criteria in LGBM, CATBoost and XGBoost (it was very unstable).
- Ensamble LGBM with CATBoost, XGBoost and NN models (results were nearly the same as LGBM standalone model has but training time increased a lot).
	
### Thanks for reading and good luck in future competitions!
