# 7th Place Solution

Competition: icr-identify-age-related-conditions
Rank: #7
Source: https://www.kaggle.com/c/icr-identify-age-related-conditions/discussion/431028

After doing few submission in initial part of the Competition, I didn't work on it, as heavy shakeup was expected.
But its a positive shakeup for me now (with a solo gold) 😅.

Here my submission details:
- Fill Nan data with 0
- 5 fold Multi Label Stratified using Greeks Values
- EJ was categorical so used Label Encoding
- Also Label Encoding for Beta, Gamma and Delta
- Used MultiClass CatBoost Classifier for all the models

Saw that Beta, Gamma and Delta have very high predictive capability, but were given only for training data,  so used all the other features to predict encoded Beta, Gamma and Delta using 5fold strategy, and used those features along with other given features to predict Multi Class Alpha. Then converted the Alpha probs to binary probs by adding B, D and G prob to predict class 1 and using A prob to predict class 0

Thats it.

Thank You

PS [here my submission code](https://www.kaggle.com/code/manthanbhagat/simple-baseline-add-greeks-features/notebook)
