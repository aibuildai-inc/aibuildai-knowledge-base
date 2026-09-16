# 1st Place Solution

Competition: home-credit-default-risk
Rank: #1
Source: https://www.kaggle.com/c/home-credit-default-risk/discussion/64821

Here we'll try to make a more or less "technical" presentation of our work. For more personal and other insights, please refer to [Olivier][1]'s [Ryan][2]'s and [mine][3] threads, as well as to [Phil][4]'s comment.

**Introduction.**

From my prior experience with credit underwriting, I’ve come to appreciate that this is one of the most complex problems for applying machine learning to. Data in this domain tends to be very heterogeneous, collected over different time frames, and coming from many different sources that may change and alter in midst of the data collection process. Coming up with a proper target variable is also a very tricky process, that requires deep domain knowledge and refined business analysis skills. I want, again, to commend Home Credit and Kaggle for coming up with such a great dataset, that was leak-free and very amenable to Machine Learning techniques. 

Based on what is known about Credit Underwriting, and in general these kinds of Machine Learning problems, it has been clear all along that there would be two things crucial for building a good model for this competition: 1. Good set of smart features. 2. Diverse set of base algorithms. We were able to have four main sources of feature diversity, and a few minor additional ones. 

**Data and Feature Engineering**

The first big set of features that we used were the features that can be found in many kernels, including excellent one by Olivier. These features employed various forms of aggregation on many-to-one tables, and joined them with application features. There were about 700 of features total that were used.

Feature Engineering (Ryan)

Like many people I started my base models with simple aggregates over SK_ID_PREV &amp; SK_ID_BUREAU features for each SK_ID_CURR. I also created a lot of features based on division and subtraction from the application_train.csv. The most notable division was by EXT_SOURCE_3 which gave me small boosts to my CV and translated positively to the LB. I also found pretty decent boosts by using a label encoder for categorical variables. I did this for all the categorical variables on application_train.csv and all the categorical variables for the LAST application in the previous_application.csv.
Aside from aggregate features over SK_ID_PREV &amp; SK_ID_BUREAU, I also used difference slices of data to compute aggregates. 

Previous_application.csv - aggregates for last 3, 5 and first 2, 4 applications. Each of these choices were tested against my CV and gave the best boost. 
Installment_payments.csv - aggregates for the last 2, 3, 5 payments. Aggregates over NUM_INSTALMENT_NUMBER at 1, 2, 3, 4 installments. Aggregates over last 60, 90, 180 &amp; 365 days filtered on DAYS_INSTALMENT. I also did aggregates over all installments that were past due. Where past due was defined as a 1 if the difference between DAYS_ENTRY_PAYMENT and DAYS_INSTALMENT was positive and 0 if not.
POS_CASH_balance.csv, credit_card_balance.csv - similar methodology as installment_payments.csv
I also used lag features from previous_application.csv. I used lag features up to the last 5 applications for each SK_ID_CURR. 

(Olivier) 
I suppose most of the competitors know my features and dataset :) Out of the public kernel I tried to compute a yearly interest rate that became one of the highest scoring feature on my models. I also had a try at creating predictions on a few tables (bureau and previous application) but for some reason I did not get the boost reported by other teams.
With new skilled members entering the team with features/dataset that gave a real boost to our CV/LB I stopped working on features and focused on stacking.

(Phil)
The most important features that I engineered, in descending order of importance (measured by gain in the LGBM model), were the following:
neighbors_target_mean_500: The mean TARGET value of the 500 closest neighbors of each row, where each neighborhood was defined by the three external sources and the credit/annuity ratio.
region_id: The REGION_ID_POPULATION field treated as a categorical rather than as a numeric feature. 
debt_credit_ratio_None: grouped by SK_ID_CURR, the sum of all credit debt (AMT_CREDIT_SUM_DEBT) over the sum of all credit (AM_CREDIT_SUM).
credit_annuity_ratio: AMT_CREDIT / AMT_ANNUITY
prev_PRODUCT_COMBINATION: PRODUCT_COMBINATION value form most recent previous application.
DAYS_CREDIT_mean: grouped by SK_ID_CURR, the mean CREDIT_DAYS value from the bureau table.
credit_goods_price_ratio: AMT_CREDIT / AMT_GOODS_PRICE
last_active_DAYS_CREDIT: From the active loans in bureau, the most recent DAYS CREDIT value, grouped by SK_ID_CURR.
credit_downpayment: AMT_GOOD_PRICE - AMT_CREDIT
AGE_INT: int(DAYS_BIRTH / -365)
installment_payment_ratio_1000_mean_mean: Looking only at installment payments where DAYS_INSTALLMENT&gt;-1000, take the mean of AMT_PAYMENT - AMT_INSTALMENT, grouped first by SK_ID_PREV and then by SK_ID_CURR.
annuity_to_max_installment_ratio: AMT_ANNUITY / (maximum installment from the installments_payments table, grouped by SK_ID_CURR).

(Yang)

My idea for the special features: some come from the open solution last 3,5,10 credit card, installment, and pos. But I modified the time period to include more variance in the features. Also, I apply the weighted mean( use the time period as the weight) to create some features related to the annuity, credit, and payment. I think these features are quite useful to extract the person credit habits. The last part I think that is interesting and useful: I generate some KPI constructed by income, payment and time which have some good impact on the CV.

Feature selection and reduction (Bojan)

One of the things that proved very useful for this competition was a good way of restricting the feature set. Various ways of aggregating features would usually result in feature sets that numbered in thousands, and it’s very likely that many, if not most, of those features were redundant, noisy, or both. I tried some very simple ways of reducing the number of features using frequency-encoded categorical features with the numerical features, and then running very simple forward feature selection using just Ridge regression. I’ve used this technique in the past for metafeatures, but this was the first time I tried it on raw features. Surprisingly enough, it worked remarkably well. I was able to reduce the “big” original set of features that numbered over 1600 features to just 240 features or so. Later on when Olivier added more features to the “base set”, I just directly added these features to my 240, and ended up with 287. These 287 features were able to give us models with CV of 0.7985 or so, with LB scores of 0.802-0.803. As we were adding more members to the team, it became important to try to combine their features with ours. It took some heroic effort on Olivier’s part to discern which one of Phil’s features were complementary to our own, and the combined set of features numbered in about 700+. When Ryan and Yang joined the team, it became impossible to repeat most of that effort. We’d try to just roughly see which ones of our features were different, and add them to theirs. 

Towards the end, it became obvious that Yang’s base set of features was perhaps overall the best. One of his Kernels alone was able to get 0.802x private / 0.803x public, thus placing it in the top 20 on the final LB. We would combine his base set with all the “special” features from the other feature sets, but due to time constraints we had to do this in a crude and probably very inefficient way. Our final “supersets” numbered in 1800-2000 features. It’s hard to put down the exact numbers - we had some “joint” sets, but we were also individually combining them for our own models. 


**Base Models**

All of our base models were trained on StratifiedKFold, with 5-folds. I choose that setup for one of my models early on, for no particular reason, and that become the default. I know there has been lots of back and forth about relative merits of Stratified vs. “regular” KFold, but in the end I don’t think that choice makes much of a difference, at least not for this competition and this kind of problems. 

(Olivier) I used LightGBM, FastRGF and had a try at FFM but CV results were really below expectations (0.76 AUC).

(Bojan) I used XGBoost, LightGBM, CatBoost, and even a simple Linear Regression. I essentially used just one set of hyperparameters for XGB (a pretty “standard” one, that you can find in most kernels, and about three different sets for LightGBMs - one that you can get in the kernels, one that the Neptune guys used, and one that’s more or less “standard” set of hyperparameters. Most of my XGB models were trained with gpu_hist on a GPU, while LightGBMs were trained on CPU. CatBoost models were not that good, and took forever to train, but I think they helped a bit with metafeature diversity. 

(Ryan and Yang) Trained several LightGBM models on their engineered datasets, as well as on some combination of their data with the rest of the datasets. Ryan also tried his luck with FFM, but no luck there either. 

Neural Nets etc. (Michael Jahrer)
As many of you have read in the discussion forum neural net models are behind boosted trees (lgbm,xgb) in terms of AUC. This is my outcome too, at the beginning it was not possible for me to reach 0.785 CV with nn, while lgbm was above 0.79. NN is always a hot candidate for blending when it comes to a certain level of accuracy, so I tried my best.
My features at the beginning was very bad, that's not my strength. After teaming up I had access to excellent feature sets. Kudos to all 5 remaining team members. Results were similar across all datasets, DAE+NN was consistently better than plain NN (maybe +0.005 in AUC), till end I never tried plain nn alone.
DAE means denoising autoencoder preprocessing as input to nn. An unsupervised technique for better data representation. Same technique as I described in porto seguro. The number of raw features are in the 100..5000 range, this means the DAE need to be very big to avoid information loss and keep the overcomplete/relaxed representation. First DAE models was always with 10000-10000-10000 topology, this means I blow up number of features to 30k. Supervised nn had always 1000-1000 topology, that's my standard recommendation, works here as well. Tried more neurons, AUC goes down.
DAE: swapnoise=0.2, high start learnrate in DAE (close to divergence), 1000 epochs. Supervised nn: lRate=2.5e-5, dropout=0.5, 50 epochs approx (until overfitting), logloss optimized. All hidden units are ReLU, Optimizer SGD, minibatchsize=128, small lRateDecay. One complete run with 5CV is about 1 day on a GTX1080Ti, DAE dominates runtime.
Raw data normalization is still rankGauss, missing values are replaced by 0. I added reference code for this normalization to my github repo.
Later in the competition I tried fewer+larger hidden layers in DAE and I guess its better here. Best AUC I got with a DAE with just 1 hidden layer with 50k neurons, followed by a 1000-1000 supervised nn (CV=0.794961 public=0.80059 private=0.79433) on a joint featureset from all 6 of us.
Apart from all the neural net optimizations a simple lgbm with small learning_rate beat them all. Best lgbm was 0.8039 on CV, nearly 0.01 higher AUC than the nn.
Anyways, I think neural nets here plays a minor role. My guess is that data normalization is still the biggest issue why they come not to same AUC level compared to lgbm.
But they were needed at the end to fight for the last 0.0001 boost, that is what kaggle is about.


**Ensembling**
For ensembling we used 3 levels with a complete pipeline of csv generation for easy sharing of meta files across team members and centralize the concatenation work.

Everyday as we got new base predictions, L1 dense matrices were produced and input to 1st level stackers : NN (guess who), XGBoost, LightGBM and a Hill Climber linear model. We ended up with 90+ base predictions there. As the number of base predictions grew, LB and CV became more correlated. I strongly believe Michael’s NNs even if they individually did not perform as good as boosters di help a lot in getting CV and LB relation more stable. 

After a few days our CV/LB stopped progressing as much as we wanted and it became clear we had to go to Level 2. 

The L2 layer contained a NN, an ExtraTree and a Hill Climber. This is where Silogram suggested we should add a bit of restacking with raw feature.

The final prediction was an equal weighted blend of these 3 predictions.

(Michael)

We had a neural net also as successful stacker. Here I used plain supervised nn with one hidden layer 500 ReLU units. The trick here was to find proper start lRate=1e-3 and good lRateDecay=0.96 (the lRate multiplier after each epoch) values to make a smooth run. Also dropout=0.3 was important here.

(Phil)
The ExtraTrees L3 model was a very shallow (max_depth=4), highly regularized (min_samples_leaf=1000) model that used just 7 of the L2 models plus a single raw feature, AMT_INCOME_TOTAL. It scored CV: 0.80665, LB:80842, PB: 80565.

**Other things that we tried, some interesting insights, and a few recommendations**

As briefly mentioned above, overall as a team we did not invest much time in hyperparamter tuning. I tried running one optimization script that Olivier had prepared, on both the XGB parameters as well as on LightGBM, but local results were somewhat discouraging. By then we were relying heavily on our ensmebles, so in our mind having several different models trained on different hyperparameters made up for not having a single, highly optimized model. 

One thing that Phil discovered, and I think there has also been some mention of this in the forums, is that it’s possible to predict train/test with over 0.98 AUC! We discovered this towards the end, but were not able to make any use of it. (Some of us were convinced that some of the top teams were in fact exploiting some kind of adversarial validation/ pseudolabeling.)

One of the things that we tried was to develop a predictive model to impute the EXT_* features. The imputed features had AUC of 0.78x, but they did not help our base models, either for CV or on public LB. Based on our analysis, we believe that might be because these features are amongst the most consistent between the train and test sets, thus not really contributing much to the “lift” that the public LB had over the local CV.

After the competition ended, we took a closer look at some of our base models. It turns out that our top three base models would all have placed in top 10, and their simple average would also been sufficient for first place! This, to me at least, suggests that for this kind of competition and this kind of problem, feature engineering and feature selection are still the most important step. Since we have not optimized our best feature set, or the hyperparametes for models trained on it, it is conceivable that it’s possible to create a single model that would outperform all of our ensembles. 


  [1]: https://www.kaggle.com/c/home-credit-default-risk/discussion/64541
  [2]: https://www.kaggle.com/c/home-credit-default-risk/discussion/64490
  [3]: https://www.kaggle.com/c/home-credit-default-risk/discussion/64480
  [4]: https://www.kaggle.com/c/home-credit-default-risk/discussion/64480#378401
