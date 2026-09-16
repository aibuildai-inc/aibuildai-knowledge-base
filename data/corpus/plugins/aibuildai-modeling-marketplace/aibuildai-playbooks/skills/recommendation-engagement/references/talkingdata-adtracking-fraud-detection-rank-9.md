# 9th place

Competition: talkingdata-adtracking-fraud-detection
Rank: #9
Source: https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56279

Congrats to all the winners, and disappointed to see so many people affected by the late sharing of scripts - so sad to see a 0.9811 sub uploaded on last day and affect people's hard work.   
   
Our solution was a couple of LGB's at public 0.9821 or 0.9820, and nnet at public 0.9816. We did a lot of feature engineering in `data.table` this is very nice for building features on such a dataset, especially, time based features. Entropy helped - [linky][1]  ; particularly entropy over time based features like minute. Device `3032` was all, or nearly all, no downloads and was not in test, so we got circa 0.0004 from removing. We also got split second times and incorporated that into lead times - this was done by turning the ordering per second, and count of clicks per second, into sub-second time - if there are 100 clicks in a second `16:00:00`, first click gets `16:00:00.00`, second gets `16:00:00.01`, next `16:00:00.02` ... etc. This gave us good lift, around 0.001 early in the competition, but we did not test the lift later to see the drop by leaving it out. Stacking gave us, circa .0003. Counts of previous periods, eg. period this time yesterday, helped also. 
A lot of other things tried with no, or small lift - used about 30 features in all.  


  [1]: https://github.com/owenzhang/kaggle-avito/blob/a7a2cc853b0ca86f07cdb9dd483779b2927b99ee/avito_utils.R
