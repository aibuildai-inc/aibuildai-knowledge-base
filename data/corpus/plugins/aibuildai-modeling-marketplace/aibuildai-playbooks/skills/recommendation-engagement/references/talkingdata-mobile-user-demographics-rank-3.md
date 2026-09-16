# Some of the 3rd place solution history (Danijel|Matias)

Competition: talkingdata-mobile-user-demographics
Rank: #3
Source: https://www.kaggle.com/c/talkingdata-mobile-user-demographics/discussion/23465

Hello, here I would like to share what we were doing all those weeks, and specially the last and leaky end of the last week. I'll give my view point and Danijel may add his viewpoint as well.

As usual in any competition, I started doing some exploratory analysis and my initial assumptions were that the ratio of usage of different apps would be predictive for age and gender. For example if the user has a lot of events associated to PokemonGo, war craft and other games, it's very likely that he is one of my coworkers! I even built this Kaggle script where I try to analyze ratio of apps type usage: https://www.kaggle.com/chechir/talkingdata-mobile-user-demographics/only-another-exploratory-analysis 

At the same time I saw the great script of Yibo: (https://www.kaggle.com/yibochen/talkingdata-mobile-user-demographics/xgboost-in-r-2-27217) and I copied his way to encode everything to 1/0 (event the ratios). Then I started to use a bunch of xgb models as well as a glmnet model, blending them all using multivariate regression (nnet package in R). I was doing reasonable well (around 20-30 place on the LB) when I saw the dune_dweller script (don't need to add the link!). At that time I was trying to learn Keras, so I used her feature engineering and plugged a keras model.. It had a great performance and boosted my score around the 17th position! For some reason I decided to share it on the Kaggle scripts: https://www.kaggle.com/chechir/talkingdata-mobile-user-demographics/keras-on-labels-and-brands. And our best model single model for devices with events is just that model with some new features and a more layers and regularization. It scored 2.23452 on the LB. 

The additional features to this model were:

 - TF-IDF of brand and model (for devices without events) 
 - TF-IDF of brand, model and labels (for devices with events)  
 - Frequency of brands and model names (that one produced a small but clear
   improvement)  

And the parameters were: 

    model = Sequential()
    model.add(Dropout(0.4, input_shape=(num_columns,)))
    model.add(Dense(n))
    model.add(PReLU())
    model.add(Dropout(0.30))
    model.add(Dense(50, init='normal', activation='tanh'))
    model.add(PReLU())
    model.add(Dropout(0.20))
    model.add(Dense(12, init='normal', activation='softmax'))

Using the average test predictions for each fold model helped a lot here (I was using 10 folds)

Then we merged our teams with Danijel who was doing a very creative xgboost combination besides other things that he would explain much better than me. Together we started retraining some of our models on CV10 and then switched back to CV5 because of processing time reasons. For the ensemble weights we used the optim package in R (iterated 3 times) and also built differen ensembles for devices with events and devices without events (that was another great idea from Danijel). 

When the leak issue raised we were around 11th to 13th on the LB and we started to look where it was. My team mate Danijel was the man how built a clever matching script that combined with our best submission allowed us to fight for the top places in those crazy 3 last days. We found also that devices with events weren't taking advantage from the leak, so we only used leak models on non-events devices. 

To me this competition was a great experience, I learned a lot from my team mate Danijel,  and also from Dune_Dweller, Yibo and from fakeplastictrees and the dam leak!
