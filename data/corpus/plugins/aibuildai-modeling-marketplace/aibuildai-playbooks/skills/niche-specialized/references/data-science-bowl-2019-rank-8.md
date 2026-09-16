# 8th place solution

Competition: data-science-bowl-2019
Rank: #8
Source: https://www.kaggle.com/c/data-science-bowl-2019/discussion/127285

congratulations to all winning teams (either winning medels or knowledge and fun). Many thanks to organizers for providing interesting data and execellent platform, also tons of thanks to generous kagglers for their sharing in disscusion and kernels. Waking up and konwing I had my first solo gold as my first medal in kaggle is just.. too good to be true😃 

### **Model**

My model is pretty simple. 3 layer mlp with 256-256-256 topology, BN and 0,3 dropout rate, everywhere. 3 leaky relu activation  + 1 linear. That’s it.
Besides accuracy _group, I use 3 times sqrt of accuracy as another target, hoping it can provide more information than just 0123 values and reduce overfitting, but seem like it don’t have very large effect on the score. 

### **Validation method**

5 groupkfold, mainly watch inversely weighted oof qwk, but also not weighted oof qwk. inversely weighted is like I described in discussion. I am not very sure if it perfectly mimic test data like truncating, but it runs fast. I had some disalignment among lb and cv. I guess they might just align at ~0.01 level. So I am pretty lucky

### **Preprocess**

Log transform and then std transform on numeric features. 
Impute missing with zeros and encode missings of features into 0,1 as new features.  

### **Training**

The best score obtaibed by using private data. I gave it a bet since private data could give 2 times amount of data and NN is data hungry. I know its distribution is different, but distribution of trainning data is also different from truncated test anyway. 

Submission A uses both private and trainning data (0.559 private lb)
Submission B uses trainning data (0.552 private lb)

Training 9 models with all data with different seeds and slightly different epochs (63, 65, 68) 

Adam optimizer, 128 batchsize, 0.0003 LR with cyclic decay:
```
def lr_decay(index_):
    if index_ &lt; 15:
        return 0.0003
    elif  index_ &lt; 30:
        if  index_ % 2 ==0:
            return 0.00008
        else:
            return 0.0002                    
    elif  index_ &lt; 40:
        if  index_ % 2 ==0:
            return 0.00008
        else:
            return 0.00003          
    else:
        return 0.00003
```

### **Postprocess**

Simple average 18 of predictions of 9 models (2 outputs, acc and acc_group per model). then use threshold Optimizer to find thresholds. I randomly initiallized the thresholds for threshold Optimizer around training target distribution, and ran threshold Optimizer 25 times, then chose the one with best cv qwk.

I did a 5 fold simulation(4 folds act as oof we have, 1 fold acts as label of test data) to compare several ways of deciding thresholds. Found that using threshold Optimizer is better than deciding thresholds by simple using training target distributiion.

### **Features**

I generated ~1100 features, and selected 216 according null importance by using rf mode in lgbm, introduced by @ogrellier in his great notebook [here](https://www.kaggle.com/ogrellier/feature-selection-with-null-importances). I found that use ~100 features gave better cv score(~0.563) than 216 features(0.559), but also low training loss and larger valid-trainning loss gap, which might indicated larger overfit. And 216 feature version have better score on LB. I chose to use 216 features  in both final submissions. 

Main feature list:

- Type, title, event counting,

-  event_id counting, 

- title_acc, title_acc_lasttime

-  title duration max/mean/std (I cliped title duration at 1000, I think 16mins is already quite long for a kid to play a session. Those duration  outlier might be errors in recording. Anyway I don’t think a kid can play a session for 3 hours),
title_misses mean/std, 

- title_round_misses_mean_divided_by_round_duration(reflect acc vs speed infomation),

- nunique_title, 

- nunique_title_in_this_world(world reflect certain facet of kid’s ability, like knowledge in length, knowledge in speed, etc)

- session_sum, event_sum, 

- game_tried_ratio(# game with try devided by # game), event_4070_ratio(# 4070 devided by # events)

- title_distraction_mean.( basically is like what I did in my previous [notebook](https://www.kaggle.com/zgzjnbzl/visualizing-distraction-and-misclicking). I count the all kids’s 4070 events and their coordinates in title_heatmaps. The inverse of counting of 4070 events on heatmaps on a certain position is the distraction-score of this 4070 event.  I assume that misclicks happens in small regions around target object, and  distraction could happen everywhere)

- Binning assessment_title counting and accuracy max ino 0 and 1(ever played vs never played, ever passed VS never passed).

- If_skip(binray features indicated if skip into this assessment from title in not designed order )

- If_repeat(binray features indicated if last session was also this assessment)

### **Some other thoughts

I fixed the memory issues of preprocessing private data at very very last time, and submit it 8 hours before competition deadline and it was running in submission for 6 hours. I made my solution literally 2 hours before the deadline.  yes people are always saying don't give up too early and now I believe it😅

### code:

resubmitted kernel here: 
[https://www.kaggle.com/zgzjnbzl/dsb-mlp-216-feature](https://www.kaggle.com/zgzjnbzl/dsb-mlp-216-feature)
