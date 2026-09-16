# 8th solution

Competition: favorita-grocery-sales-forecasting
Rank: #8
Source: https://www.kaggle.com/c/favorita-grocery-sales-forecasting/discussion/47564

First of all, I'd like to thank Giba, my team mate, without whom I would not have entered this competition.  I also want to thank Kaggle and the host for providing this challenging dataset.  Congrats to all gold winning teams and individuals, you did an awesome job here.  And there is lots of solution sharing already, which is great.  Last but not least, final standing is a great, positive, surprise to us, esp considering we both had the flu.

Our solution is mostly a single NN model that scores 0.515 private and 0.508 public, with a secondary lgb model.  Believe it or not, we missed the deadline by few seconds and could not select our second sub ourselves.  It scored 0.514 on private LB, maybe we would have advanced a couple of place, but we are very happy with the result anyway.  Details below.

**Missing values**

onpromotion is fully known after 2014-03-31.  We ignored the missing values issue by only using data posterior to that.  We actually did not use data prior December 1st 2015.  We also assumed that any item/store that appears somewhere in train or test data has 0 sales for the dates it does not appear.  This seems to be what the data description says.  It means that store/item that do not appear in train have zero sales during the train period.

**Seasonality**

In time series problems there are two key ingredients: seasonality, and cv setting.  Here weekly seasonality is extremely strong, yearly seasonality is weak, and there is a monthly seasonality due to pay day.  We dealt with weekly seasonality by using time periods that align on weekdays for everything.  We did not find a good way to use the monthly seasonality.  And we captured the yearly seasonality by using sales figures from 364 days ago.  We used 364 and not 365 to align with weekdays.

**CV setting**

We selected 2 validation periods plus the test period:  (2017-07-16, 2017-07-31), (2017-08-01, 2017-08-15), and (2017-08-16, 2017-08-31).  For each period we construct a series of training datasets aligned on weekday, for instance for the first validation period:

    dataset up to 2017-03-11 predict period of length 16 starting on 2017-03-12
    dataset up to 2017-03-18 predict period of length 16 starting on 2017-03-19
    dataset up to 2017-03-25 predict period of length 16 starting on 2017-03-26
    dataset up to 2017-04-01 predict period of length 16 starting on 2017-04-02
    dataset up to 2017-04-08 predict period of length 16 starting on 2017-04-09
    dataset up to 2017-04-15 predict period of length 16 starting on 2017-04-16
    dataset up to 2017-04-22 predict period of length 16 starting on 2017-04-23
    dataset up to 2017-04-29 predict period of length 16 starting on 2017-04-30
    dataset up to 2017-05-06 predict period of length 16 starting on 2017-05-07
    dataset up to 2017-05-13 predict period of length 16 starting on 2017-05-14
    dataset up to 2017-05-20 predict period of length 16 starting on 2017-05-21
    dataset up to 2017-05-27 predict period of length 16 starting on 2017-05-28
    dataset up to 2017-06-03 predict period of length 16 starting on 2017-06-04
    dataset up to 2017-06-10 predict period of length 16 starting on 2017-06-11
    dataset up to 2017-06-17 predict period of length 16 starting on 2017-06-18
    dataset up to 2017-06-24 predict period of length 16 starting on 2017-06-25
    dataset up to 2017-07-15 predict period of length 16 starting on 2017-07-16

We use all datasets but the last to train, and we use the last one for early stopping.  We use 10 fold CV on the training periods.  The only caveat is to make sure all store/item pairs for a given time series are in the same fold.  I used a very similar CV setting in WTF competition.

**Feature engineering**

All unit sales are clipped to be non negative, then log1p transformed. For each of the above datasets, we use:

 - lags over periods similar to @Ceshine Lee starter LGB kernel.  We use mean, max, and proportion of zero entries, as well as proportion of promotion days.

 - Sales for each of the last 7 days

 - Average sales per weekday over the last 8 weeks

 - For each of the test/validation days: sales from 364 days earlier, and promotion status.

 - Class of the item

 - Store

We did not use oil price, item number, and other info in our primary model.

**NN Model**

A feedforward model with 3 dense layers, relu activation.  Class and store are embedded in 4 dimension vectors, and appended at the second level.  Let x be the output of the dense layers.  Its length is equal to the length of the test or validation period.  Last level is a 1D convolution with x,  the on promotion input vector, and the product of the onpromotion vector with x:

    y = Multiply()([x, promo_test_input])
    y = Reshape((-1,1))(y)
    x = Reshape((-1,1))(x)
    z = Reshape((-1,1))(promo_test_input)
    x = Concatenate(axis=-1)([x, y, z])
    x = Conv1D(1,1, activation='linear')(x)

This captures the influence of onpromotion day per day quite well.
The target for the model is the sequence of train/validation sales, i.e. 15 or 16 days depending on the period.

**Other models**

We trained a lgb model using the same features and cv setting, the only difference being the target: we train one model per day, like in @Ceshine Lee starter LGB kernel.  Best lgb scores 0.510 on pblic LB and 0.517 on private LB.  I don't know how others got better results with LGB than with NN.

Giba created a classifier model to predict when sales are zero or not. Alone this model is useless but it adds when ensembling.  I'll let him describe it if need be.

**Ensembling**

Turns out that what would have been our best sub is an unweighted average of 3 different runs of the NN model using a different sets of training periods, and one lgb model.  We tried some limited stacking with a NN with one hidden layer.  Giba also looked to how to post process submissions.  By default we set to 0 all store/item pairs that do not appear in train, but he found that we should not do it if that pair has a promotion in test.  He got a sub that scores 0.514 private that way, but, due to few second glitch we did not not select it...  And we did not have time to combine his postprocessing with our best blend either.

**Conclusion**

Main takeaway from me is once again to ignore public LB feedback.  Indeed, that feedback was on the first 5 days of the test period, and it gives zero useful information on what happens for days further in the future.  Having some reasonable CV setting was key to estimate how well a model predicts further in the future.  I could not say I was expecting to get a result as good as what we got in the end, but I knew we were not overfitting to the first few days of the test period ;)  

Let me end with a special notice to Ceshine Lee without whom many would not have fared as well as they did.  He shared a great kernel, and he shared very useful tips in the early days of the competition.  There should be a sharing gold medal in Kaggle competitions, and he would have won that one here!

**Edit:** 
The discussion below makes me add this.  We added 0 sales row with onpromotion=False for all store/item pairs that appear in our training datasets.  We did not add all those store/items pairs that appear in the test set.  We did compare with and without these: the results were similar, but training time were higher with the test items.  We validated adding rows with onpromotion=False by using the original data in the validation periods.
