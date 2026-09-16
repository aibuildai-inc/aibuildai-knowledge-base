# 10th Place Method : Moderate? Stacking

Competition: tabular-playground-series-sep-2021
Rank: #10
Source: https://www.kaggle.com/c/tabular-playground-series-sep-2021/discussion/276009

A big thank you to Kaggle and all discussion and code contributors!

I haven't done ML competitions since about 3 years so I wanted to brush up and see what new tools and techniques have since emerged. After such a long time, I didn't recognize anyone from the old times in the discussions other than @tunguz . I was hoping to finish top 10 and I had many short sleeping nights :) It was a joy finding out I reached my personal goal for this playground competition.  

The short story is that I prefer the purism of a single model, but in the end, I succumbed to stacking.

# Details

## 1.  Starting point
First I wanted to keep things simple and work with single models. I was planning to select the best single model and focus in just one of them once it became clear which one was the most promising. I tried the ones I was familiar with: XGBoost, LightGBM, CatBoost, SVM, and Keras.

I've never had much luck with NNs in past competitions for tabular data. It was my goal to see if I could reverse that too by trying methods I read in the past month.
 
## 2. Single model designs
All my single model designs had only one engineered feature: the sum of NA's in the row. It was clearly effective from discussions and it was well understood why it works from past competitions. I did try some other engineered features but I decided to just go with NA's in the interest of time. 

Tuning. I believe Bayesian optimization was what people used 3 years back. I learned optuna this time. I don't know which one performs better.  

[This Keras kernel](https://www.kaggle.com/lukaszborecki/tps-09-nn) by @lukaszborecki was inspiring. It has a simple but not trivial layer structure and scored well on the leaderboard. All my Keras kernels were derivative of this one. His kernel had embedding layers with 96 binned inputs and a separate input without binning. I learned about the "swish" activation. 3 years back I'd use "selu","elu" or "relu".
 
Keras didn't give me scores anywhere close to the gradient boost ones. However, I knew they are many times effective when stacking with gradient boost components. I was by this time convinced I had to succumb to stacking because the single models were not scoring competitively at the leader board. 

## 3. Stacking 
The stage was ready for stacking. I had the same [single level stacking structure](https://www.kaggle.com/c/tabular-playground-series-sep-2021/discussion/275740) as @vkonstantakos. I had 17 base level models: 5 LightGBM, 4 XGBoost, 2 CatBoost and 6 Keras.
The stacking method was LogisticRegression. I did try XGBoost for the stacking but it wasn't performing well. LogisticRegression also gives how much weight each of its input component is being used so it's easier to tell which base models to keep.

[This kernel](https://www.kaggle.com/mlanhenke/tps-09-simple-blend-stacking-xgb-lgbm-catb) by @mlanhenke was influencial for me. His xgb1 component caught my attention. It had a max_level of 18, it didn't perform well as a single component but it seemed crucial for the overall stacking. It seems to be doing  the job of adding a strong uncorrelated component. 

## 4. Conclusions
It was definitely a challenging enough competition for a brush up. I wish I had gone a bit further with feature engineering and representation learning in Keras though. Having a better insight into how to design good uncorrelated components like xgb1 is something to keep in mind. Those would be in my todo list!
