# 15th Place Solution for the "ICR - Identifying Age-Related Conditions"

Competition: icr-identify-age-related-conditions
Rank: #15
Source: https://www.kaggle.com/c/icr-identify-age-related-conditions/discussion/431415

I feel incredibly fortunate to have reached 15th place in the competition. I started by referencing the familiar baseline(https://www.kaggle.com/code/aikhmelnytskyy/public-krni-pdi-with-two-additional-models). By delving into the comprehensive notebooks and discussions, I learned and adapted my approach.

What did not work for me:
- Avoided using Epsilon from greeks due to potential data drift.
- De-anonymizing (But it's still a magic method for me who is not good at math, thanks!) and feature derivation.
- Feature selecture using target permutation, I found that the features filtered by "gain" and "split" are not in good agreement.
- Optuna, postprocessing, and oversampling, they weren't effective for me.

What did work for me:
- Simulate a local verification process through Nested k-folds with StratifiedKFold. Not the best_model but full models of cv_outer.
- A diverse ensemble model with probability reweighting.
- Incorporating the greeks.Alpha into the training.

I'm very lucky this time (contrary to real life...orz). My deepest gratitude goes out to the members of the Kaggle community who generously and selflessly share their knowledge. I also hold immense respect for those inquisitive minds who are never hesitant to ask questions and challenge the status quo. Kindly bear with any oversights or shortcomings!!
