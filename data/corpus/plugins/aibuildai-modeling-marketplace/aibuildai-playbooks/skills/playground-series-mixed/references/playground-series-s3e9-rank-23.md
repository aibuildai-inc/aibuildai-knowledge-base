# #12th place solutions:  My 6 step process for any competition

Competition: playground-series-s3e9
Rank: #23
Source: https://www.kaggle.com/c/playground-series-s3e9/discussion/394600

I wanted to detail my process that I created to be reapplied for each competition or problem.

It is a 6 step/ notebook process which includes all code linked: 
1. [EDA ](https://www.kaggle.com/code/slythe/pse3e9-1-eda-model-selection-regression) 
2. [Feature Creation ](https://www.kaggle.com/code/slythe/pse3e9-2-feature-creation-symbolic-transform)
3. [Feature Selection ](https://www.kaggle.com/code/slythe/ps3e9-3-feature-selection-regression)(RFECV and manual selection) 
4. [Single Model Tuning](https://www.kaggle.com/code/slythe/pss3e9-4-single-model-tuning-regression/notebook)
5. [Multi Model CV ](https://www.kaggle.com/code/slythe/pse3e9-5-2-multi-model-cv-all-folds-regression/notebook)  / [Multi Model : best fold ](https://www.kaggle.com/code/slythe/pss3e9-5-1-multi-model-cv-best-fold-regression/notebook)
6.  [Ensembling](https://www.kaggle.com/code/slythe/pss3e09-6-ensembling-regression/notebook)

The first and foremost is an investigation of the data, I prefer doing this myself even well into the competition with hundreds of EDAs, as this gives me insights into the data without being  influence by others

* **Step 2** is a separate notebook just for **feature creation**, I use a number of automated and intuitive processes here. My favorite is [Symbolic Regression Transformation](https://gplearn.readthedocs.io/en/stable/index.html) which tries to apply mathematical operations to the features that explain the target. Very interesting but it actually didn't work that well this episode (it did well in episode 5) 
* **Step 3** is a feature selection notebook where I take all the features created in step 2 and run recursive feature elimination on them (manually and automatically using [sklearn's REFCV](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html))
* **Step 4** is self explanatory, I added the features created previously and try optimize the model. I dont like Optuna as it tends to overfit and try do this manually.
* **Step 5** I take every Tuned model and run a nested CV as well as best-in-fold CV. Similar to step 4 just with many models. 
**Note:** I try to save all the validation predictions and test predictions for later use. Also optimizing the validation predictions give a good indication of the models performance (NB dont use the training predictions for this --although I have to for [Multi Model : best fold ](https://www.kaggle.com/code/slythe/pss3e9-5-1-multi-model-cv-best-fold-regression/notebook)
* **Step 6** is to apply ensembling techniques to the outputs from all the above test and validation predictions. Scipy Optimize seems to do very well here as well as calibration using linear models 

**#### What I did differently this competition ####**
I tested additional features found in other top notebooks and they didn't seem to improve my CV --but I think I was wrong here

 What I eventually used was to group the duplicates in the trainset by each column and get a range of target values 
Features are then created from these target values by using them as bins 

The idea here is that the duplicates have differing targets so why not use the target mean for each grouping 

```python
def Additional_Features(df_in):
    df = df_in.copy(deep = True)
    
    for col in df_test.drop("is_generated",axis =1).columns: 
        grp_target = df_trn[df_trn.drop(target,axis =1).duplicated()].groupby(col).mean()[target]
        if 0 in grp_target.index:
            bins = list(grp_target.index)+ [max(df_trn[col])+1]
        else: 
            bins = [0] + list(grp_target.index)
        
        #add cols
        df[f"{col}_grp_mean"] = pd.cut(df[col], bins=bins, labels =grp_target.values )
        df[f"{col}_grp_mean"] = df[f"{col}_grp_mean"].astype('float64').fillna(0)

    return df

df_trn = Additional_Features(df_trn)
df_tst = Additional_Features(df_tst)
df_trn
```

This improved my linear models tremendously from approx. 14 RMSE to 12 

Well that's it, I hope these notebooks help someone and can be used in future competitions. Please let me know if you see any references to your code as I only recently made them public
