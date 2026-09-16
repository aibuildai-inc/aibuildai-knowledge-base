# [3rd Place] Solution

Competition: ashrae-energy-prediction
Rank: #3
Source: https://www.kaggle.com/c/ashrae-energy-prediction/discussion/124984

Thank you to Kaggle and ASHRAE for hosting this competition

My solution is likely to disappoint some of you given the lack of sophistication. 

I spent most of my time (too much) on trying feature engineering, trying to look a cv split correlated to public LB (didn’t happen), trying to find a neural network architecture that will give me a boost in local cv and public LB (didn’t happen) and browsing research papers related energy prediction in the subway.  

I am quite clueless about web scraping but thanks to @gunesevitan for showing the way and teaching me a lot in the process nevertheless. 

Given diverse experiments with different CV schemes I did over the period of the competition, I decided to simply combine all the results (over 30), I got into a single submission using a simple average after selection by pearson correlation (6th on private LB). 
In two instances over the last 3 weeks, I used a subset without leak of these experiments and use the leak to ensemble them but I gave up because the public LB was quite poor despite a better local CV and I thought it was overfitting. However, as I selected this method as my alternative solution, it appears this was the best on the final private LB.


# Preprocessing:

By lack of time, I only used the ideas and code of some excellent public kernels. 

I also wrote a script that run for a few hours where I eliminated all 0s in the same period when these 0s occurs in the same site, at the same period and across all meters. @ganfear wrote an excellent visualization [here](https://www.kaggle.com/ganfear/missing-data-and-zeros-visualized). My goal was to eliminate a maximum of "vertical" lines especially if they were simultaneous. I believe that this is the trick that gave me an advantage. Using this preprocessed data in the top public kernels always gave me a better LB, so I was on the right track.


# Feature Engineering:

-	Row feature from train/test, weather metadata, and building metadata
-	Count features and combination of features (see excellent write up on IEEE competition)
-	Lag on row features but only the temperature feature lags seemed to be actually useful.
-	Features mentioned in public kernel such as RH, feel_likes and presence of other meters seems to have a marginal upside.
-	A feature that I only included in the other submission in the last days found in a research paper was working well with meter==1 was :

```
latitude_dict = {0 :28.5383,
1 :50.9097,
2 :33.4255,
3 :38.9072,
4 :37.8715,
5 :50.9097,
6 :40.7128,
7 :45.4215,
8 :28.5383,
9 :30.2672,
10 :40.10677,
11 :45.4215,
12 :53.3498,
13 :44.9375,
14 :38.0293,
15: 40.7128,}

train_df['latitude'] = train_df['site_id'].map(latitude_dict)
train_df['solarHour'] = (train_df['hour']-12)*15 # to be removed
train_df['solarDec'] = -23.45*np.cos(np.deg2rad(360*(train_df['doy']+10)/365)) # to be removed
train_df['horizsolar'] = np.cos(np.deg2rad(train_df['solarHour']))*np.cos(np.deg2rad(train_df['solarDec']))*np.cos(np.deg2rad(train_df['latitude'])) + np.sin(np.deg2rad(train_df['solarDec']))*np.sin(np.deg2rad(train_df['latitude']))

train_df['horizsolar'] = train_df['horizsolar'].apply(lambda x: 0 if x &lt;0 else x)
```
it is supposed to calculate the solar horizontal radiation coming into the building.
 
 
# Models:

I trained Keras CNN (@aerdem4 style), LightGBM and Catboost on diverse version of cleaned data, various feature selection (including removal of building_id) - without ever beating the best kernel without leak to my disappointment. 

I had better success on local CV with meter level model using decision trees rather than using all the meter in the same decision trees model. Catboost and Lightgbm are clearly using different selection of features for the splits, so they are quite complementary.

Only towards the end of the competition, I realized that the NN was giving approximately the same results on both local cv and public LB than decision trees models and that most of the performance was due to the cleaning more than my feature engineering. However, the results were quite uncorrelated, so it was good for diversity.


# Ensembling:

My best submission is an ensemble of a small samples of leak free (~10) predictions among all my experiments. Using lightgbm, I used these different predictions as feature with the addition of the original feature “meter” in order to reduce overfitting. The leaky rows are used as training set in the ensembling model.

My idea was that it is likely that the prediction of first 11% of the test set (included in the public LB) are going to be much better than the last 11% of the test set (private LB), so if I were to use the leak on the test set, I may be able to correct the last 11%. I was expecting a small hit on public LB but my public LB took such a larger hit than expected (+.025) that I thought I was overfitting and didn’t pursue that path. I kept the submission as my second final submission which became my best on private LB.



# Didn’t work for me or didn’t give any significative improvement:

Pseudo labeling (read @hengck23 posts for more information)
NN decoder-encoder using the leak in @mjahrer style (reniew seems to have succeed in this direction: see here (https://www.kaggle.com/c/ashrae-energy-prediction/discussion/122203#699375)
Thermometer encoding on cloud coverage
Treat temperature features as categorical variables (given limited number of values) and use target encoding.
Adaptation of @anokas  tool for time series feature
…

Otherwise, I would like to thank all the people who taught me a lot on Kaggle among whom @Raddar, @cpmpml and @titericz during the years.


Update:
github code available [here](https://github.com/chabir/kaggle_materials/tree/master/ASHRAE%20-%20Great%20Energy%20Predictor%20III%20-%203rd%7C3614)
