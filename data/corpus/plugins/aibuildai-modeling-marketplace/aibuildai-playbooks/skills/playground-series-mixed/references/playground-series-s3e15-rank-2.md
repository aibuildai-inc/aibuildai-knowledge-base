# 2nd place solution explained: the power of original dataset

Competition: playground-series-s3e15
Rank: #2
Source: https://www.kaggle.com/c/playground-series-s3e15/discussion/413826

First of all, I would like to express my gratitude to Kaggle team for hosting this competition. It's a great opportunity for beginners to get our hands dirty with some close-to-real-world data and get some practise. As someone who embarked on this data science journey just 5 months ago, coming from an entirely unrelated field, it holds significant meaning for me to see my progress and achievements in this competition.

Earlier I have made a post regarding [feature range comparison](https://www.kaggle.com/competitions/playground-series-s3e15/discussion/413293) between original and competition datasets. My solution is based on those observations and **my plan was as follows**:

1) Do a brief EDA of competition dataset to gain initial understanding of data. 
1.1. `'D_h'` and `'D_e'` are highly correlated and are mostly the same (influenced my imputation technique).
1.2. There was class imbalance, `'tube'` geometry is the majority class (influenced choice of model).
1.3. Data is missing at random, which is probably the effect of artificially adding noise (influenced both choice of model and imputation strategy).

2) I decided to go with gradient boosting only, as it can handle both missing values and class imbalance quite well. In the previous imputation competition winning solutions were based on deep learning, but I believed that because of smaller size of the dataset and large amount of missing values it wasn't worth trying. I created a few baseline models and filled the missing features statistically (filled NaNs with mean of each feature grouped by `'author'`) to check its performance.

2.1. FLAML single LGBM model gave CV of 0.0733 (used @paddykb FLAML [notebook](https://www.kaggle.com/code/paddykb/ps-s3e14-flaml-bfi-be-bop-a-blueberry-do-dah) from 3.14 competition with few modifications, thanks for the great framework).
2.2. Optuna-based baseline also gave CV around 0.073 (thanks @tetsutani for this [model](https://www.kaggle.com/code/tetsutani/ps3e15-eda-ensemble-and-stacking-baseline), I used some code from that notebook).

3) CV scores of baseline models looked very good already, so instead of spending time on finding a better model, I decided to work on better imputation technique and finding some "tricks", knowing this data is synthetically generated.

3.1. As I already saw the dataset had a lot of added noise, I decided to compare it with original dataset. I have shared the insights earlier [in this post](https://www.kaggle.com/competitions/playground-series-s3e15/discussion/413293).
3.2. After I discovered that some features always go together and are unique to a single author, I imputed the competition data by finding the unique pairs and triplets in original dataset.
3.3. Some features were still missing, so I have decided to try different MICE-based packages like *miceforest*, *missingpy*, etc. Also, there was a [great notebook](https://www.kaggle.com/code/arunklenin/ps3e15-iterative-catboost-imputer-ensemble) by @arunklenin where he used custom implementation of CatBoost-based MICE algorithm, which also performed very good. 
3.4. I have also tried to round the imputed features to the closest unique from original data, but this made the CV worse. Also thanks @alexdippolito for this [topic](https://www.kaggle.com/competitions/playground-series-s3e15/discussion/412509). He also explored features from original dataset, which made me feel I am on a right path. Also congratulations on getting the third place!

4) My final model uses the ensemble of LGBM and XGBoost, and scores were as follows: 


I also tried some experimental models, the best one was the blend of two ensembles, which were trained on competition data, imputed using the method above, and oversampled original data. However, its CV was so close to the one for model above and there was a chance that this model would score too low on private leaderboard, so I decided the improvement wasn't better than random and I didn't select it (but actually, it would give me the first place). 


**What did I learn from this competition?**
- **Spend more time on EDA and knowing your data.** Many notebooks were having a lot of visualisations without any explanations. 
- **Experiment more and always trust your CV.**  Always rely on your CV, use public leaderboard as one of the tools, but don't rely on it. Also don't avoid checking notebooks with lower score, as they may have more useful information than top notebooks.
- **Use the fact that Playground datasets are synthetic.** By exploring the original data you will most likely come up with some tricks which make your final score a lot better.
- **Don't be shy as a beginner.** I was hesitant at first, as it's been only 5 months since I started learning, but it turned out my chances to get the top score were just the same as everyone else's. Also Kaggle community is super friendly and always helps. 

Thanks to all Kagglers who are sharing notebooks and ideas! Good luck with further competitions!
