# 20 Private LB rank solution

Competition: ashrae-energy-prediction
Rank: #20
Source: https://www.kaggle.com/c/ashrae-energy-prediction/discussion/123654

It was a long run and finally it is completed and we can share some tricks and ideas that we have used. According to this official [discussion](https://www.kaggle.com/c/ashrae-energy-prediction/discussion/123462) and this [Kernel](https://www.kaggle.com/robikscube/ashrae-leaderboard-and-shake) [ods.ai]PowerRangers has taken 20 Private LB rank. So let's start :)

Sorry for all misspelling and not really good code style in Kernels and here too ( LGBT-&gt; LGBM, Bland-&gt; Blend, ...). We did not have enough time to write really good code base :( 

# Our team had 2 main obstacles:
- Time (The exams were coming)
- Computing resources 
As for computing resources - we carried out all our experiments and development in Kaggle Kernels. So we had to deal with RAM and execution time limitations.
# Our main scheme:
.png?generation=1577617664090195&amp;alt=media)


# First of all, we created some Baseline models:
1. We started from statistics - simple mean or median over each `meter`. Then same statistics over several categorical features ( `meter`, `day_of_week`, `building_id`, `month` ). You can find it in this [here](https://www.kaggle.com/vladimirsydor/naivemeanpredictor) `NaiveMeanModel` . Of course, they were performing really poor (1.39-1.4 on public LB). But we used these feature for more sophisticated models.
2. Then we tried [RandomForestRegressor](https://www.kaggle.com/vladimirsydor/randomforestbaseline). But it was pretty slow and perform not really good.
3. Finally we realized that 

## So we made our first experiments with LGBM and passed into silver zone

# And now it was time for PREPROCESSING!
[](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1690820%2Fe958aa1905d44ec6a5dd083a2b4e95e5%2FData%20Science%20Tom%20and%20Jerry.jpg?generation=1577608845429010&amp;alt=media)

## Preprocessing:
- First off all - detecting outliers in nearly the same way, as in this [discussion](https://www.kaggle.com/c/ashrae-energy-prediction/discussion/122471), but less sophisticated.
- Weather data preprocessing - interpolation on NaNs and creating features `is_nan` for these columns 
- Adding features: `day`, `day_of_week`, `month`,  rolling features ( `{feature}_mean_lag{window}` ), max/min features by categorical features ( `air_temperature_max`)
- Finally we tried to add Leak data from 2017-2018 but it was not a good idea for us. But adding leak data from 2016 gave us some more data for training

## You can have more detailed look at our Data Preprocessing and Feature engineering:
- [Preprocessing](https://www.kaggle.com/vladimirsydor/baseline-preprocessing)
- [Preprocessing + Feature Engineering + Leaks from 2016](https://www.kaggle.com/vladimirsydor/baseline-preprocessing-leaks-train-fe)
- [Preprocessing + Leaks from 2016-2018 )](https://www.kaggle.com/vladimirsydor/baseline-preprocessing-leaks) . Here we created several datasets, taking leaks from different sites in order to train uncorrelated models.

Also most ideas ( and code :) ) were taken from these kernels:
- [first](https://www.kaggle.com/purist1024/ashrae-simple-data-cleanup-lb-1-08-no-leaks)
- The second one was deleted

# Now was time for modelling!


# Models:
## 1. Of course, LGBM:
-  We are nor really professionals in LGBM hyperparams optimization + it was training really long, so we did not spend a lot of time for hyperparams optimization. But what we find out is that it was not overfitting much and increasing the number of leaves mostly helped the model (we tried 145 and 82)
- Even with high LR our Boost has not converged even for 7k iterations. But Leak scores and Public LB scores did not really improve for Boost trained for 7k, comparing with boost on 5k. Also it was a great difference for Boost trained for 3k. We could not try more iterations, cause Kernel has time limitations :(
- Also we had 5 kernels for one Booost in order to train it in CV style and then blend results.
- One more interesting fact is that - all DataFrames that passed to LGBM model are converted to `float64` and if you have real BIG DATA, you will RUN OUT OF MEMORY, so you need to convert it into `np.array`. Small tip but it helped us a lot :)
You can take a look at our Boost models here:
- [with leaked data from train(2016)](https://www.kaggle.com/vladimirsydor/lgbt-on-pp-leaks-train-fe-fold-1)
- [with leaked data from train(2016) and test(2017-2018)](https://www.kaggle.com/vladimirsydor/lgbt-on-leaks-fold-1)
- [without leaked data](https://www.kaggle.com/vladimirsydor/lgbt-fold-1)

## 2. Neural Net ( going out from forest!!! )
[Kernel](https://www.kaggle.com/vladimirsydor/nn-on-pp-leaks-train-fe-fold-1)
Inspired by [abazdyrev](https://www.kaggle.com/abazdyrev) and his [Kernel](https://www.kaggle.com/abazdyrev/energy-consumption-keras-approach)
-  First of all, usual preprocessing for NN: scaling and label encoding for Embeddings. You can find it in  `PreprocessingUnit`
-  So we have chosen Embeddings for categorical features, because we have really a lot of data and they could train ( I hope )
- Also we tried several optimizers (Nadam, Adam, Adamax) and as for final activations ( no-activation and softplus). Best results were for softplus and Adamax. We did not try to submit NN predictions alone but on Leak Validation it even outperformed LGBM ( WOW!!! ).

# And finallly we are ready for the most interesting part - BLEEEEEEEEEND


We were inspired by this [kernel](https://www.kaggle.com/khoongweihao/ashrae-leak-validation-bruteforce-heuristic-search) , but it was too simple for us :)

[Kernel](https://www.kaggle.com/vladimirsydor/bland-by-leak)
1. Firstly we gathered a lot of submissions  ( our and some public ones ), finally we had 32 .csv files
2. Some public kernels and some our submissions were created with models trained on Leak data , so we can not use them for Leak Validation and they were excluded. Also we exclude some bad submissions. You can find out them in `EXCLUDE_LIST`
3. Then we have to choose several submissions for BLENDING:
- Firstly, we tried Hyperopt on indexes for median blending. We have taken 10 submissions. Also we penalized Hyperopt for taking same files for blending 
- Secondly, we have created some kind of Genetic Algorithm for the same purpose. Mostly it was taken from our practical work from the university.
- Thirdly, we tried to stack submissions with one layer Perceptron in CV style but it was hardly overfitting, so we did not try it more.


# Finally all that Leaked data was added to our final submission.  
[Kernel](https://www.kaggle.com/vladimirsydor/add-leak) 

# Also add some less valuable Kernels to build the complete scheme:
- [Blending NN trained on different Folds](https://www.kaggle.com/vladimirsydor/bland-nn-on-pp-leaks-train-fe)
- [Blend LGBM (PP + Leaks Train + FE) on different folds ](https://www.kaggle.com/vladimirsydor/bland-lgbt-on-pp-leaks-train-fe)
- [Blend LGBM on different folds ](https://www.kaggle.com/vladimirsydor/bland-lgbt-folds)
- [Blend LGBM (trained on train + leaked data) on different folds ](https://www.kaggle.com/vladimirsydor/bland-lgbt-on-leaks)
- [Leak Aggregator ](https://www.kaggle.com/vladimirsydor/leakaggregator)

#Great thanks to my team members:
 [Evgeniy](https://www.kaggle.com/zekamrozek) and [Vladislav](https://www.kaggle.com/vladyelisieiev). They made a real good job!!!

## Good Kaggling and happy Holidays !!!
