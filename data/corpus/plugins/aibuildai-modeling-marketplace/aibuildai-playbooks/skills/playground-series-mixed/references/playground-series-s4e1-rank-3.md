# 3rd Place Solution: CatBoost Encoding Galore

Competition: playground-series-s4e1
Rank: #3
Source: https://www.kaggle.com/c/playground-series-s4e1/discussion/472413

Welp, I have finally reached top 3 in Playground Series after 11 months. I never imagined that my first competition of this year would end in such an insane way. After all, getting 3rd place out of 3600 teams is something that's on much higher level than what I've achieved last year.

So without further ado, here's what I've done.

# Feature Engineering

1. I applied TF-IDF vectorization on some features and then decompose the result with TruncatedSVD (credit to @arunklenin). However, unlike most people, I tried to build a class so I can implement it within a pipeline and do different type of vectorizations and decompositions on different set of features.
2. I created new features based on [this notebook](https://www.kaggle.com/code/aspillai/bank-churn-catboost-0-89626) by @aspillai. However, scaling and `IsSenior` features aren't included. I also modified `Sun_Geo_Gend_Sal` for my own purposes. I also will refer to it as `AllCat` from now on.
3. I created a new feature named `ZeroBalance` as indicator whether a customer has zero balance or not.
4. Finally, I casted both `EstimatedSalary` and `Age` as integer by multiplying them by 100 and 10 respectively first. Why? You will see it the main reason soon. However, one funny side effect is that just multiplying the Age by 10 will give you a boost when paired with the code for binning from @aspillai notebook (it's equivalent of binning `Age` by dividing it by only 2).

# Encoding

Before we're getting into ensembling, I want to talk about encoding first, which is the key to getting high performance. There are three types of encoders I used in this competition: CatBoost's built-in encoder, CatBoost Encoder from `category-encoders` library, and M-Estimate Encoder. The first one is obvious why, but the second one is because I also wanted to use apply CatBoost encoding on other estimators. As for the third, it's because XGBoost and LightGBM doesn't like too much CatBoost encoding. I also won't explain much about it as it's not the main contributor to the high performance.

Now, let's get into what features I had encoded. Actually, let me rephrase the sentence. **Let's get into what features I did not encode**. In the original set of features, there are only `Balance` and `HasCrCard` as the unencoded features. The rest? Pretty much all encoded. This includes float features such as `EstimatedSalary` and `Age`, and now you know why I casted them as integer. Also, do you remember that I referred to one of feature engineering as `AllCat`? That's because I concatted almost all features I planned to encode in that feature, with exception of `IsActiveMember` because I also encoded `IsActive_by_CreditCard`. In total, there are 12 features I had encoded... or so you thought.

Remember TF-IDF vectorization and SVD decomposition? Well, you can encode them too! Just do the same thing as what I had done to `EstimatedSalary` and `Age` to the decomposition result. As a note, I only did encoding on SVD decomposition of `Surname` with 4 components, even though I also did vectorization and decomposition on `AllCat` and some other features.

Another important thing about the encoding here is that, CatBoost encoding actually cares about the order of your dataset so much. Well, not by default but, you can set it as such. In fact, `category-encoders` CatBoostEncoder treats different orders of data differently. In order for CatBoost to disallow permutation of dataset when encoding features, you have to set `has_time` parameter to `True`. And the best order of the dataset? When concatting the original dataset and the competition training dataset, you have to put original dataset **before** the competition dataset. This will give you the best result for this competition.

# Ensembling

I used 7 models on this competition.

1. Logistic Regression is the lowest performing model but also the greatest one for experimenting which features you need to encode with `category-encoders` CatBoostEncoder. In a way, this can function as indicator on which features you need to encode for CatBoost.
2. For Neural Network, I used Input -> 32 LeakyReLU -> 64 LeakyReLU -> 16 LeakyReLU -> 4 LeakyReLU -> 1 Sigmoid architecture, with AdamW optimizers. It has the same encoding as both Logistic Regression. Also, this is the only model where I didn't apply any vectorization.
3. XGBoost has both CatBoost Encoder and M-Estimate Encoder for different set of features, and I used Optuna for HPO. Vectorization is also applied to 4 features with 500 max features and 3 decomposition components.
4. LightGBM is somewhat similar to XGBoost when it comes to pre-processing.
5. 3 CatBoost with different bootstrap type each: no bootstrap, Bayesian, and Bernoulli, with same exact preprocessing: vectorization on 2 features with 1000 max features and 4 decomposition components as one of them. All of them have +0.902 CV score. I didn't do any HPO because they're already as slow as snail.

The weights are defined with Ridge Classifier. And if you're curious, I implemented all preprocessing within each model pipelines because I'm a freak when it comes to leakage preventation in cross-validation.

# Additional Note

1. One way to boost the score from what I have noticed is to increase the number of folds. Therefore, I used 5-folds for experimentation and 30-folds for submission, which almost took 12 hours.
2. One thing I wished to discover and do earlier was to do multiple concatenation on the same original dataset, as concatenating twice actually gave me the best private LB score (maybe this is the black magic that the top 2 had done idk).
3. I also applied postprocessing related to the data leakage by @paddykb.
4. Finally, you can read [my notebook here](https://www.kaggle.com/code/iqbalsyahakbar/3rd-place-solution).

Thank you everyone who has participated in this competition. I hope you will learn a lot from this write-up. Also, I will try to participate in the discussion next time as I have no more reason to stay silent after getting the prize (unless I have a very big chance on reaching 1st) :)
