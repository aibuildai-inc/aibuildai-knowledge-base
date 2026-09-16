# 5th Place Solution | Imputation Without Any Imputers

Competition: playground-series-s3e15
Rank: #5
Source: https://www.kaggle.com/c/playground-series-s3e15/discussion/413742

Well... the shakeup is bigger than I thought, and I certainly didn't expect that my gamble has paid off... or sort. Also I've finally reached top 5 after all this time joining Playground Series!

So if you read the title at first, you might be thinking that it's impossible. However, that's exactly what I did: **I wrote my own code to impute all features except `x_e_out [-]`**.

# Imputation

A few days ago @shalfey posted an interesting stuff about the characteristics of original dataset, such as how each `author` strictly only has one `geometry`. The thing is, I found the exact same thing **one week earlier**, with only Pandas Dataframe's `groupby` method. Here are most of the characteristics that I've found:

1. Each author has their own unique geometry
2. For some authors, there is only one unique `pressure [MPa]`
3. For some authors, there is only one unique pair of `D_e [mm]` and `D_h [mm]`
4. For some authors, there is only one unique `length [mm]`
5. For each pair of `chf_exp [MW/m2]` and `length [mm]`, there is only one unique `geometry`
6. Each `D_h [mm]` has exactly one unique `geometry`
7. Each `D_h [mm]` has exactly one unique `D_e [mm]`

My imputation is basically finding the missing value based on the value of other features in original dataset. For example, in order to impute `D_e [mm]`, I have to use available `D_h [mm]` value on the competition dataset and use it to search `D_e [mm]` on the original dataset. 

Now, some of you have probably known that since we're using synthesized dataset, making some relationship between features get messed up and creating noises in the dataset. Obviously this will make competition dataset different from original dataset. However, this isn't a problem. If you can't find the value in the original dataset, **I just need to find the closest value to it**. This is also how I impute features that need more than one other features, such as relationship 5. For example, if I want to impute `geometry` based on `chf_exp [MW/m2]` and `length [mm]`, I have to find all unique values of `length [mm]` in original dataset that is related to the value of `chf_exp [MW/m2]` on the competition dataset, then find the closest unique values to `length [mm]` on the competition dataset. Only after that, I try to find `geometry` based on the `chf_exp [MW/m2]` and `length [mm]` unique values that we've found.

These are a lot. However, it's only enough to almost fully impute both `geometry` and `D_e [mm]`. In order to impute the rest, I have to do some **improvisation**, such as imputing `D_h [mm]` based on `D_e [mm]`, imputing `author` based on `length [mm]`, imputing geometry based on `chf_exp [MW/m2]` and `length [mm]`, etc.  I also have to repeat some imputations to get rid of the missing values. These aren't fully accurate to the original dataset of course, but our competition dataset has messed up relationship with noises anyway, so it doesn't matter.

The full code and almost fully imputed dataset is available [here](https://www.kaggle.com/code/iqbalsyahakbar/ps3e15-major-imputation) if you want to read it.

# Feature Engineering

Some of my feature engineering ideas are based on cylinder. I was inspired to do this based on the existence of length and diameter features. My ideas are as follows:

1. Difference between heated and hydraulic diameter
2. Cylinder surface area and its difference between heated and hydraulic version
3. Cylinder volume and its difference between heated and hydraulic version

There is also one other idea. It's based on how `x_e_out [-]` doesn't have any metric, as you can see from its feature name. The idea is to do some simple math operations on `pressure [MPa]`, `mass_flux [kg/m2-s]`, and `chf_exp [MW/m2]` in order to get rid of their metrics.

The effect of those ideas are... mixed to be honest. Some models got improvement, some got worse. However, it didn't matter as I use different subset of feature engineering ideas to build my ensemble anyway.

# Models and Ensemble

This is nothing special to be honest, and it's probably the cause of me failing to get higher position. I use different subset of feature engineering ideas to creat different models, use different estimators on some of them. In order to find the optimal weight, I use Ridge regression, allowing both interceptor and negative coefficient to be fitted. I also use original dataset to train my models (I don't include it on my validation of course). Finally, in order to tune my models, I use Optuna for Gradient Boosting such as XGBoost and LightGBM, and use manual tuning for everything else. That's it, there is not much else to explain here.

The full code of my model building and EDA is available [here](https://www.kaggle.com/code/iqbalsyahakbar/5-part-2-model-building?scriptVersionId=131500687). You can also see comparison between my imputed dataset and the competition dataset here.

Thank you for reading, and I hope this will be useful to you!
