# 2nd Solution, CPMP View

Competition: ieee-fraud-detection
Rank: #2
Source: https://www.kaggle.com/c/ieee-fraud-detection/discussion/111321

When Giba and I joined the team, @sggpls @tony321 and @krivoship already had a solid position in top 10 and there were less than 10 days left.  It is hard to add value in this case, and I decided to focus on adding diversity to the models they were blending.   For that I started my own dataset and borrowed bits of theirs.  Let me describe the steps I took.

**Data Cleaning**
The data was a bit similar to Malware, with time stamped observations, and many features with lots of values present in test and not in train.  I screened all features by looking at train vs test distributions, for raw feature and also frequency encoding of features.  Here is an example for `card1`.  
[card1 plot]
Raw data is used on the left column, and feature encoding of it in the right column.  Red is train, blue is test. Left row x axis is `card1`.  Top left y axis is frequency.  Top left show frequency of each card1 value in train (red) and test (blue).  Bottom left y axis is mean of target with a log transform.  0 is the average for train.  Right column is the same, except card1 is replaced by log of its frequency.

We see that lots of card1 values only appear in test.  If we use it directly it will lead to major drop in private LB.   We also see that frequency encoding of card1 looks more balanced across train and test.  FOr this feature I discarded the original one and kept the frequency encoding.

I did not merge values as I did in Malware, for lack of time.  I am not sure it would have helped a lot, in hindsight.

**Cross Validation**

Cross validation for time series is always tricky.  Here is what I ended up with.  I tried to mimic the fact that there is a significant time gap between train and test, and also that test can be several months away from train.  

 I run these folds, using months starting with with 0.  The | indicates the train/val split
    0 | 2 3 4 5 6
    0 1 | 3 4 5 6
    0 1 2 | 4 5 6
    0 1 2 3 | 5 6
This makes me train 4 models for a given model type.  I then average the auc score of each of these models on the corresponding validation data.  This scheme or some variants were reused by my team mates for validating their models.  The advantage is that we have folds that test long term forecast.

I used this CV to evaluate features, to evaluate HPO, and for feature selection.   For submission I tried two ways.  One, to extrapolate the number of trees from the CV runs.  I just ran a linear regression on the log of trees, and predicted the value for full train.  I also tried to use an unshuffled 8 fold CV and average all fold model predictions for test.  Both ways were giving similar LB score.  I probably have used both and blend the result, but I didn't...

**Feature Engineering**

I started by computing frequency encoding of all features, then used lightgbm with the above cross validation to select features using [permutation importance](https://www.kaggle.com/c/ieee-fraud-detection/discussion/107877#latest-635386) with some modification: I kept a feature only if permuting it did not improve any of the 4 model predictions.  This removed some features.  I repeated the process, using permutation importance to remove features until the feature list does not shrink anymore.  With this I got 0.942 on public LB.

The next level was to look for user id, and this is when I teamed.  I was hoping my new team mates had sorted out the user id puzzle.  They did.  They actually had a number of different uids.  I selected two of them defined as follows:

    data['uid1']` =  (data.day - data.D1).astype(str) +'_' + \
                data.P_emaildomain.astype(str)
    data['uid2'] =  (data.card1.astype(str) +'_' + \
                data.addr1.astype(str) +'_' + \
                (data.day - data.D1).astype(str) +'_' + \
                data.P_emaildomain.astype(str))

where `data.day` is the number of days since the beginning, computed from `TransactionDT`.

I did not want to use these user ids directly for fear of overfitting (in hindsight this was a mistake, and in my last sub I just ran catboost using these as features and it led to my best sub).  I rather used these uids to compute aggregate features from the sequence of time intervals for transactions for a given uid, and also for teh transaction amount.  Here is one way I used:

    def add_gr(data, col):
        cols = data.columns
        gr = data.groupby(col)
        data[col+'_count'] = gr.TransactionID.transform('count').astype('int32')
        data[col+'_next_dt'] = gr.TransactionDT.shift(-1)
        data[col+'_next_dt'] -= data.TransactionDT
        data[col+'_mean_dt'] = gr[col+'_next_dt'].transform('mean').astype('float32')
        data[col+'_std_dt'] = gr[col+'_next_dt'].transform('std').astype('float32')
        data[col+'_median_dt'] = gr[col+'_next_dt'].transform('median').astype('float32')
    
        data[col+'_next_amt'] = gr.TransactionAmt.shift(-1)
        data[col+'_mean_amt'] = gr.TransactionAmt.transform('mean').astype('float32')
        data[col+'_std_amt'] = gr.TransactionAmt.transform('std').astype('float32')
        data[col+'_median_amt'] = gr.TransactionAmt.transform('median').astype('float32')
        new_cols = list(set(data.columns) - set(cols) -set([col+'_next_dt']))
        return new_cols

where col is one of `uid1` or `uid2`.

With these I got 0.952 public LB.

Another significant improvement was to use target encoding of these uids and a set of transaction chains that my team mates derieed by combining uids with additional feature columns that represent successive transaction amounts, eg `V307 `.  This moved my lgb model to 0.9606 public LB unless mistaken.

I added few other features that improved a bit, but nothing very important.


**Final Blend**

I mostly used lightgbm, but I also tried xgboost and catboost.  XGboost was giving models extremely close to lgb, with both very high correlation in predictions and similar LB score.  My features were added my  team mates already very good models, yielding even higher scores: Giba had one at 0.9644, and my Russian team mates had models up to 0.9626.  Giba then used his blending magic to reach 0.9662 public LB the day of original deadline.  We then wondered about what could yield us the extra 0.001 we needed to reach prize zone.  We realized we were not using the main property of the dataset: if a card is a fraud at some point, then it is a fraud in the future of that point.  Well Giba captured that partially with some lag variables, but that was it.  We then added some post processing that overrides predictions if a card is a fraud in the past, and this moved us to 0.9672 at the last minute.

**What did not work**

Many things, but the one I spent most time on what to try to leverage the fact that if a card is a fraud at some point, then it is a fraud in the future of that point.  I didn't want to use post processing as it can lead to overfitting if the card ids are not perfect. Indeed, if one of our id actually covers several cards, then one fraud isn't necessarily followed by all frauds.  I therefore tried this:
Create one submission.  Then for each uid create a lag variable equal to the average of predictions before the current point, and train again.  This is like stacking, but with a temporal shift.
This did not work, CV skyrocketed, but LB dropped by 0.01.  I still think there is something to be done to make it work, but I could not find how.

**What worked**

Teaming.  See [here](https://www.kaggle.com/c/ieee-fraud-detection/discussion/111159) for more details.

It is clear that 3 + 1 + 1 &gt; 5 here.  If you have never teamed then try, you will not regret it.

Edit: Additional information about our solution can be found in these two posts:

How to find card ids: https://www.kaggle.com/c/ieee-fraud-detection/discussion/111559
The whole team story: https://www.kaggle.com/c/ieee-fraud-detection/discussion/111554
