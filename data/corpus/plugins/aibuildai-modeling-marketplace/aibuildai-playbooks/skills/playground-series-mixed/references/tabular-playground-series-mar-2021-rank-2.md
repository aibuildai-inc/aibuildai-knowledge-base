# 2nd place solution - dae + embeddings

Competition: tabular-playground-series-mar-2021
Rank: #2
Source: https://www.kaggle.com/c/tabular-playground-series-mar-2021/discussion/229868

Here we go again 😄 ... 

Initially I only wanted to test some ideas so I combined the best parts of both @ryanzhang and my winning solutions from the previous competitions + added embedding inputs for categorical variables.
[[new_1c86ab636a2a3f5f6.png]](https://gifyu.com/image/Yqti)

My main idea was to use the learned embedding representation both for data (xtrain) and target (ytrain) during the representation learning phase. Just because I did not like how noise + one hot encoding fit together. 
[[new_4.png]](https://gifyu.com/image/Yqtj)

1. shows the basic idea of ohe
2. shows what happens when noise is added pre ohe
3. shows the noising part post ohe. I didn’t like the fact that this type of data is beeing fed into the DAE. Where a observation can belong to all types of level at the same time.

So I added noise to label encoded categoricals, put them through a embedding layer and reconstructed the target/ clean part on the fly.
[[new_3.png]](https://gifyu.com/image/Yqtu)

Noisy data is used as input, embeddings are beeing trained, ytrain is beeing created and voilá. I also applied this idea to the masking part. 
[[new_2.png]](https://gifyu.com/image/Yqtm)

The initial/ default mask has to be adjusted to match the embedding representation.
I didn’t do much tuning/training.
- All categoricals were put through an embedding layer of dim 5 (even the binary ones). Increasing or tuning this per categorical might improve the 1st stage model
- Fixed noise of 0.25 randomly distributed. I didn't try to increase/ decrease noise.
- DAE trained for 1000 epochs
- Quite simple two layer stage two MLP (1000r-1000r-s)

And because this worked so well I also trained a lightgbm model and stacked both lgbm + three dae runs via xgblinear. 

# Models
1x **lgbm** = **0.89743 (cv)** | **0.89304 (public lb)** | **0.89769 (private lb)**
3x **dae mlp** = **0.90042 (cv**) | **0.89560 (public lb)** | **0.90012(private lb)**
\* haven't uploaded every single dae - mlp run so only one score here.

**final xgblinear stack** = **0.9008564 (cv)** | **0.89599 (public lb)** | **0.90053 (private lb)**
