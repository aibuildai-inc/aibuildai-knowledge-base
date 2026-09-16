# 3rd Place Solution | KNN Imputation + LightGBM/XGBoost Ensemble

Competition: playground-series-s3e15
Rank: #3
Source: https://www.kaggle.com/c/playground-series-s3e15/discussion/414027

I'd like to start with a big thank you to Kaggle for running these playground competitions and everyone in this great community who has shared their valuable knowledge.  I have learned so much from participating in these competitions over the last couple months!  

Below is a summary of my approach, notebook can be found [here](https://www.kaggle.com/code/alexdippolito/playground-series-s3e15-3rd-place-solution).

**Pre-Processing**
The initial steps involved a lot of manual work going through the data and looking for patterns to solve the easier missing values.  This filled out most of the values in columns D_e, D_h, and length.  A small sample of that code for author ‘Inasaka’ rows is below

data.loc[(data['author'] == 'Inasaka') & ((data['D_h'] == 3.0) | (data['length'] == 100.0)) & (data['D_e'].isnull()), 'D_e'] = 3

data.loc[(data['author'] == 'Inasaka') & ((data['D_e'] == 3.0) | (data['length'] == 100.0)) & (data['D_h'].isnull()), 'D_h'] = 3

data.loc[(data['author'] == 'Inasaka') & ((data['D_e'] == 3.0) | (data['D_h'] == 3.0)) & (data['length'].isnull()), 'length'] = 100

My next step of pre-processing was using the K-nearest neighbors imputer to fill in the remaining holes in pressure, mass flux, and whatever was remaining in D_e, D_h, and length.  Thanks to @validmodel for [this post](https://www.kaggle.com/competitions/playground-series-s3e15/discussion/410645) which laid out many different imputation techniques. This ultimately led me to choosing sklearn's KNN imputer to finish the remaining numerical values.  I tried several different values for the n_neighbors parameter and found I was getting the best CV scores for my models in the 87 to 89 range.

The missing categorical columns were imputed with some more manual code, such as below

data.loc[(data['author'] == 'Inasaka') & ((data['D_h'] == 3.0) | (data['length'] == 100.0)) & (data['D_e'].isnull()), 'D_e'] = 3

data.loc[(data['author'] == 'Inasaka') & ((data['D_e'] == 3.0) | (data['length'] == 100.0)) & (data['D_h'].isnull()), 'D_h'] = 3

data.loc[(data['author'] == 'Inasaka') & ((data['D_e'] == 3.0) | (data['D_h'] == 3.0)) & (data['length'].isnull()), 'length'] = 100

This did reasonably well and I was getting solid CV scores but not the strongest public LB scores (more on this later).

However, near the end of the competition @shalfey made this [excellent post](https://www.kaggle.com/competitions/playground-series-s3e15/discussion/413293).  I placed all of this code to the front of my pre-processing.  It led to a noticeable improvement in my scores and I most certainly would not have finished quite as high as I did without it.

**Model Building and Cross Validation**
I first built a basic untuned XGBoost model to get a baseline CV score.  I was very surprised when it came in at around 0.07305.  That score was much better than the 1st place score of the public LB (around 0.0743 at the time).  This caused me a sense of dread 😬.

It was extremely unlikely that this basic model would not only put me in 1st but put me in 1st by a huge margin.  I continued forward anyways and tuned the model up, made some test predictions, and submitted.  The public score came to 0.07555.

There were two possible reasons for the gap between my CV score and the public LB.
1.	My CV was flawed somewhere (I strongly suspected that my imputation pre-processing strategy had caused target leakage)
2.	The 20% of data used for the public LB contained an abnormal section of data that my model didn’t do great on but would perform closer to my CV over the entire test set.

I was pretty sure the problem was number 1.  I spent a lot of time changing up my workflow to eliminate any possibility of target leakage.  However, my CV scores were still coming out to around 0.0732.  I was also scoring much worse on my public LB scores when submitting those models.  These results were starting to convince me that reason number 2 was actually true.

Based on this, I went back to my original workflow of doing all pre-processing and imputation ahead of my cross validation instead of doing it fold by fold.  In this case, there was a tradeoff between slight target leakage (since the KNN Imputer utilized the target feature) and increasing the data quality/accuracy of input features for model training.

If I was setting up a real world cross validation experiment the correct choice would be to do all imputing on a fold by fold basis and gather more data if necessary to improve results.  However, in this case, gathering more data is obviously not an option.  I was limited to the competition/original datasets and needed to squeeze every bit of information out of them to maximize the accuracy of the KNN Imputer step.  Even if doing so caused a tiny bit of target leakage.

**Utilizing the Original Dataset**
One unique feature of doing these playground series competitions is that all of them use synthetic data generated from an original dataset.  This always leads to the question “What do we do with the original data?”  Based on the playground competitions I have participated in, the answer is that you always need to find a way to utilize it.

I have to credit @adaubas for [this post](https://www.kaggle.com/competitions/playground-series-s3e14/discussion/409242) during the blueberry yield competition for really getting me to think about how the original dataset should be used.  I used to just treat it as simply additional data and immediately join it to the train dataset. But, I have found that is not the best way to handle it.

The problem with doing this is that it will change the distribution of your OOF validation data when doing CV.  The synthetic data seems to always be much noisier than the original data.  [This post](https://www.kaggle.com/competitions/playground-series-s3e14/discussion/409417) from @sergiosaharovskiy is a great visualization of the observed phenomenon.  Including the original data in the OOF validation data will give you a better CV score than what you can expect when using your model to predict purely synthetic test data.

I have found the best way to use the original data is to keep it separated from your train data and then concat the entire original data back to each fold in your cross validation.  Example code below where X_original and y_original are the entire original dataset.

`  
    
    kf = KFold(n_splits=10, random_state=8, shuffle=True)

    for train_idx, val_idx in kf.split(X_tr, y_tr):
        X_t, X_val = X_tr.iloc[train_idx], X_tr.iloc[val_idx]
        y_t, y_val = y_tr.iloc[train_idx], y_tr.iloc[val_idx]
        
        X_train = pd.concat([X_t, X_original], ignore_index = True)
        y_train = pd.concat([y_t, y_original], ignore_index = True)

        model = LGBMRegressor()
        
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        
        y_pred = model.predict(X_val)
        
        score = mean_squared_error(y_val, y_pred, squared=False)
        
`

**Models and Ensemble**
I ended up tuning 4 models that I was happy with and wanted to see how they would perform as an ensemble.

Model 1: LightGBM using all features of the original dataset except for geometry

Model 2: LightGBM same as model 1 but also added two new features ‘adiabatic_surface_area’ and ‘surface_diameter_ratio’.  Credit to @tetsutani and [his notebook](https://www.kaggle.com/code/tetsutani/ps3e15-eda-ensemble-and-stacking-baseline) for the feature ideas

Model 3: LightGBM same as model 2 but also added a single feature/component using PLSRegression on features mass_flux, pressure, and chf_exp.  Once again thanks to @adaubas for [this post](https://www.kaggle.com/competitions/playground-series-s3e14/discussion/409242) which led me to learning about PLS.

Model 4: XGBoost same features as model 1

I had originally planned to explore @tetsutani’s [notebook](https://www.kaggle.com/code/tetsutani/ps3e15-eda-ensemble-and-stacking-baseline) more and try out LAD regression and hill climbing from @samuelcortinhas [notebook here](https://www.kaggle.com/code/samuelcortinhas/ps-s3e3-hill-climbing-like-a-gm).  Unfortunately, I simply ran out of time and my ensemble ended up just being 4 basic weights for each model.

**Conclusion**
One thing that this competition further reinforced in me is that the most important thing to strive for is getting a CV workflow that you trust in.  Let your CV scores guide your decision making and don’t place too much weight on the public LB scores.

Once again, thanks to everyone in the community for being so welcoming and eager to share their knowledge!  I have learned so much! Best of luck in your future competitions!
