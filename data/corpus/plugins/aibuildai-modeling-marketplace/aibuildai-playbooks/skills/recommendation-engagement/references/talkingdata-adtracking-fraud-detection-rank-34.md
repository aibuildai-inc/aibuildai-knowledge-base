# 34th place, thoughts on CV and Target Encoding in R

Competition: talkingdata-adtracking-fraud-detection
Rank: #34
Source: https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56304

Hi!
I wanted to share some of my findings during this competition. 
It's my first ever competition or ML task, so much of what I tried was really bad, but it taught me good intuition making these horrible mistakes and have a better "radar" for possible leaks and over fitting.  

I learned a lot on R's data.table, LGBM, XGBoost, I experimented with keras and H2O NNs (didn't get good results) and also experimented with AWS's VMs.

My public lb is 65 and private is 34, i think this increase came from good generalized CV and features, after trying many things and participating in many discussions on the matter.

I ended up using 2 LGBM models (second one uses the predictions of the first, details in item #3),
with main features including: Count clicks by groups, UniqueN by groups, Next click by groups, and the target encoding based features detailed in item #3)

I won't be long on all the mistakes i did, i will focus on the things I did that gave the significant boosts along the way and seemed to have generalized well.
I hope this will be of any value to any of you.

1. Dropping any concerns for over fitting my model to day 9 hour 4, although the difference of CV and LB score was lowest, it obviously overfit.
After some discussions on the matter, I ended up training on days 7 and 8, CV with day 9, and then retraining on same rounds on entire data. 
**I read @CPMP's post and learned that he used 1.2 multiplier on the Num_Rounds with the full training set, will definitely try it in the future!!

2. I calculated all of my feature calculations on train + test supplement data.
This is after trying to run features only where hour exists in the test set, or only on the train\ regular test set.

3. My last boost up from 0.9806 to 0.9816 was using an interesting approach to target encoding i encountered somewhere in the discussions (I don't remember who suggested the approach so sorry for not giving the credit).
The method involved taking my best lgb model and predict on all train and test supplement data.
Then using the predicitions to do target encoding on the entire data and using it to create features involving the "is_attributed" field.
After some experimentation, i ended up with this:
I took the prediction and converted it 3 times to a boolean 1/0 "is_attributed" value.
- Once if pred &gt;= 0.99 then 1 otherwise 0
- then pred &gt;= 0.997 then 1 otherwise 0
- Then pred &gt;= 0.999 then 1 otherwise 0

I did this because i wanted better sensitivity for the model, i am ok missing some 1's but don't want many 0's to be mistakenly predicted as 1's.
I then calculated 2 sets of features 3 times, once for each prediction result set.
I had mean target features by groups (past is_attributed = 1 divided by count of clicks), and another set of features that included, amongst other features:

- how many times did user (ip, os, device) ever download any app

- how many times did he download the current app

- how many unique apps did he click on.

After running some test runs of the new features, i found the highest increase in score with mean target when using the &gt;=0.997 prediction, and the the other set of features maxing with the &gt;=0.99 prediction.
I then used both of the sets in a new lgb model that uses only the Quantity (non categorical) features from the first lgb model  + the new target encoded features.


4, I used bayesian optimization to optimize hyper parameters for the 2 models separately.
Unfortunately, even after testing for different methods of optimization (GP Upper bound, expected probability etc.) with different Kappa\Epsilon results. I did not find it running differently or optimizing better in any way.
It was basically running like random search.
I still need to learn more on it, but taking the best found hyper parameters did give me a decent boost in score.

Thanks for reading and thanks for being such an awesome community!!
And of course to all the awesome Kagglers that assisted and shared their wisdom during the competition.
This was quite an intense 2 months =)
