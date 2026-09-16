# genetic algorithm solution (20th place) - very long read

Competition: porto-seguro-safe-driver-prediction
Rank: #20
Source: https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44659

kernel: https://www.kaggle.com/jacekpoplawski/genetic-algorithm-main-part-20th-place/

It will be long post, but maybe someone will find it interesting - I want to show you my journey with this competition and I think my solution is quite different.

No ensembling, no OHE, no manual feature engineering. I have more than 1500 different models. I could use just 3 of them to get higher score I have now.

## Intro ##

I am a C++ programmer with over 13 years of commercial experience, I had big love for AI and Alife in 90s, but it took me some time to realize "Deep Learning" means Neural Networks. This year I started doing courses on udemy, then Andrew Ng courses in coursera. Then I found Kaggle.

It was my second competition. I have spend lots of time on Porto Seguro. In the last weeks I was sure this time was wasted, because my models were not really better than average model from public kernels. I was shocked yesterday when I realized I jumped over 1000 places up and I am 20th. I will try to describe what I tried and I what i learned. Please correct me if I am wrong somewhere.

I removed some stories related to my Neural Network approach or different variances of genetic algorithm to make it more focused and less boring ;)

I will probably publish some kernel in next days, I will try to choose single one with good private score (not sure will it be calculated for new kernels?)

## First approach - build basic subset of features ##

According to discussions on forum some features are just noise. According to xgb feature importance some features are very important. How can we check which features should be used?

I think simply testing random forest or xgboost feature importance is not a good idea. If you have some score and you will look on feature importance you should not assume that by removing some low-important feature your score will be higher. You can remove feature which is not used at all, or you can be sure that some feature is big part of your model. But features can also hurt your model and feature importance won't tell you that.

Instead we should build model with different feature subsets and compare which one has best score. How many combinations we have?

len(train_data.columns) - 2 = 57

If I am correct you can calculate it this way: each feature can be part of model or not. So it will be 2^57 minus 1 or maybe 2^56 minus 1. Anyway it's a quite big number.

We can't expect to check them all. So how to find good subset? 

We know that ps_car_13 is important. I wasn't able to run xgboost with single feature so we need at least two. Let's do following:

- take one feature

- build model from ps_car_13 and this feature

- calculate score

- repeat until all features are processed (each feature plus ps_car_13)

So now we have two features. Let's move forward and find third one. Then fourth. Etc.

Then we have n=15 features with some nice score close to 0.270. Now we can try opposite way:

- remove one feature from the model

- calculate score

- repeat until all features are processed (each model is n-1 features)

And again we can find feature which removed gives highest score.

By executing this algorithm we can find nice subset of features. Later in the discussions I found Boruta algorithm which different but quite promising too.

## Hyperparameters tuning ##

To create a model you need to execute some algorithm - like xgboost. And while this algorithm works on dataset it is also configured by many hyperparameters. In general you need make model as complex as needed but no more, because it will overfit, or as simple as possible but not more, because it won't be able to achieve good score.

I tried hyperparameter tuning multiple times and the best way in this competition I found was step by step random search. 

- start with default hyperparameters

- set random depth - random.randint(3,7) for instance

- calculate score

- very important: save hyperparameters and score to external file, write separate jupyter notebook to display scores and plot curves or scatterplots

- repeat infinite times, manually break loop when you will be happy with graph

It is very tempting to skip plotting and just look on numbers on the screen. After I started using plots I never stopped. You just see on plots what's going on, while you can be fooled by looking on raw numbers.

Depth is most important parameter because it defines how complex your tree will be. Then you need to deal with overfitting by using lambda and alpha. Now plots are extremely useful. Then you should also try to reduce number of features used in your trees and part of data. In xgboost this is colsample and subsample. There is also nice setting for unbalanced datasets - scale_pos_weight.

## LightGBM ##

I didn't know what LGB is. It was word used in discussions and kernel titles but it was similar to LB so it wasn't clear to me what's the difference between LB and LGB ;)

When I was starting with Kaggle I wanted to use NN only. Then I realized xgboost is only way to go. When I looked at some kernels I have noticed that people build xgb and lgb models and blend them. So I wanted to try it just to see will there be any difference.

When I was calculating my models with xgb I realized it uses only small part of CPU. Why? 
Maybe because it's Windows or because it can't use threads, I don't know. So I was using multiple jupyter notebooks with xgboosts and it was working. Well maybe except I needed to fit my models in memory. How to fit multiple models in memory? Don't use many features. That was main reason I stopped using One Hot Encoding.

Using lgb was very easy. I could use same code as for xgb, just changed few lines - create data for lgb instead xgb and then call train with different hyperparameters. Was the score much different? I seriously have no idea, because the first thing I've noticed was CPU usage. LightGBM is using my whole CPU and it's much much faster.

I submitted some xgb and lgb data with LB 0.280 or 0.281 and soon I stopped using xgb at all.

## Target encoding ##

My first Kaggle competition was NYC Taxi. The target was trip duration between two points on map. There was very interesting idea for feature engineering in this competition. You can do some clustering and then find average trip duration between two clusters. Or better, not just trip duration, but calculate feature called speed which is distance divided by trip duration. The first kernel which uses similar idea I saw here was Olivier's kernel with target_encode method.

At first I didn't know why he used some calculations inside this method. There is some min_samples_leaf and smoothing. Was it some magic? Well, I did what I learned previously - I plotted it. Then I understood everything.

Target encoding means that you group data with the same column value, then you calculate average target and use this average target as new column. This method can lead to overfitting when used incorrectly. For instance when number of records with given column value is very small your new column will store just target value. So algorithm will try to predict target from target. This is pointless and won't work at all with test data. So you need to be sure that number of records is big enough, that's what min_samples_leaf is. So what should you store in this new column when number of records is small? Just average value of target in whole dataset. It won't hurt. Result is just blend of average global target value and average target value for this column value.

And now the most important thing. When you open public kernels you will notice little nice method called add_noise. What it does? It does absolutely nothing. Please go to kernel sections and check yourself. Maybe I missed some kernel where it is used, or maybe I am wrong (I never copied this code, I wrote it from scratch) but if I understand correctly everyone just copied add_noise which is called with argument equal to zero.

When you are using target encoding on the same data you are doing your training you must use noise. Without noise the result will be bad. I compared noise_level = 0.1 then noise_level = 0 then noise_level = 0.5 and larger. At first it was counter intuitive. How can large noise be useful? It can. If your data is 0.1, 0.2, 0.3 and you add large random noise and result will be 0.7 0.4 0.9 - you still have 3 different values which can be used by tree algorithm, and you have no overfitting.

(this is very important to understand, I can draw some table if needed)

But now you may ask - I took one categorical value and converted it with complex algorithm into another categorical value, what's the difference? 

Because I can use more than one categorical value! I can combine them! Back to NYC Taxi. In this competition we were using 4 values (start/end trip points) for target encoding. So why not use this method in Porto Seguro?

And now the madness starts.

## feature engineering on steroids ##

How many new features can you create from some numeric data you don't understand at all? A lot.

First you can add them. 

ps_car_10_cat_add_ps_ind_02_cat means you add columns ps_car_10_cat and ps_ind_02_cat

You can also sub them, mul and divide.

Is there any point in dividing number of apples by number of oranges?

I think yes, at least it was best idea I had. In Andrew Ng course he gave example of some feature engineering where you divide two values and it can be very useful. Important note: there is one difference now between tree and NN: when you divide something you can get NaN values which are perfectly fine with tree but problematic with NN.

You can also try (x1+x2)/x1 or (x1-x2)/(x1+x2) but I had no time for that. So my math features were only add, sub, div and mul.

Then there was target encoding. My notation was like that:

ps_car_07_cat_mean_ps_ind_02_cat_mean_ps_ind_10_bin - is average target value grouped by [ps_car_07_cat, ps_ind_02_cat, ps_ind_10_bin]

ps_car_11_mean_ps_ind_10_bin - is average target value grouped by [ps_car_11, ps_ind_10_bin]

0_mean_ps_ind_04_cat - is average target value grouped by [ps_ind_04_cat]

As I said before I wasn't using OHE or any features from public kernels (I tried "missing" but it wasn't useful for me so i removed it)

So now I am able to store my features in list of strings, score them and save scores and feature list in csv files.

## genetic algorithm ##

Let's take basic set of features. For instance all features trom train set minus *calc*. Now let's generate random feature - math one or mean one (see previous section). Then let's calculate its score. Save it to the file. Repeat. Repeat. Repeat all night.

Now we have nice long csv file with many feature sets and by using some pandas and matplotlib we can see the scores. It's time for stage two.

Sort feature sets by score. Take two random sets with high score. Add sets. Randomly remove some feature or add new generated one. Calculate score. Save and repeat, repeat, repeat.
After some days of different workflows I got collection of feature sets with 0.290 score in my CV. I mean average of 5 folds was over 0.290. I got also about 500 feature sets higher than 0.288. 

## Failure ##

Then I started submitting them to LB. Score was 0.282. Once it was 0.283. Later I started blending them. 0.284 wow. So at least my idea was better than basic xgboost model I was starting with. 

https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/43455

I tried also ensembling or using rank instead average. But then I wanted to blend 300 models and it was difficult to do something more than averaging with them. I even found way to split data to 10 parts and then fit many hunderts datasets in memory to calculate rank. But it was still 0.284.

There were 0.287 scores in public kernels. 0.288 in LB. And single 0.285 models in discussions. What can I do with my poor 0.284? It was time to admit I have lost. And I wasted lots of time, both my coding time and computation time. At least I learned something, maybe?

## Grand Finale ##

5 days ago I wrote on forum:

"I have build large number of feature sets and calculated local CV for them. Then I am averaging best ones (according to local CV). In 4 days I will see was it good idea or bad. My only chance for higher position on LB is that most public kernels overfit public LB :)"
https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44210#248360

my best submission was 0.29173, submitted 9 days ago, it was average of 3 kernels, only 3, hand picked, no random hyperparameters or seeds

my other submissions: 0.29166, 0.29165, 0.29164, 0.29162, 0.29159, 0.29158, 0.29156, 0.29155, 0.29154 - none of them was selected by me to use

selected submissions were 0.29153 and 0.29147 - blend of 180 models with random hyperparameters, submitted 4 days ago, after that submission I decided I am done

## Conclusion ##

- Kaggle is the best place on the planet 

- by reading kernels and discussions and building your models at same time you can learn a lot

- it was easy for people to lose their money in tulip mania, it's easy to follow crowd on stock market, but it's good idea to make sure you know how the world works and don't overfit to LB which is just random 30% of test data

- it was very bad idea to stop fighting just before the end, my algorithm was working and with enough time I could try my feature sets with NN and xgb 

- I hope for me it is just a start - thanks for reading :)










![my top submissions - none of them selected][1]


  [1]: https://i.imgur.com/gcKlR5N.png
