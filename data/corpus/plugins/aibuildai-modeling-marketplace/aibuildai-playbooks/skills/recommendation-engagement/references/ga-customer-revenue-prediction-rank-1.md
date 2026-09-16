# Winning solution (link to kernel inside)

Competition: ga-customer-revenue-prediction
Rank: #1
Source: https://www.kaggle.com/c/ga-customer-revenue-prediction/discussion/82614

Hi Kagglers,

Key points of my solution:

**Train creation**
As we have 168 days (2018/05/01-2018/10/15) of sessions  for customers in the test, 62 days (2018/12/01-2019/01/31) of target calculation period and 46 days (16/10/2018-30/11/2018) of gap between above two windows, it’s absolutely clear to construct train data by analogy. 
I took 4 non-overlapping windows of 168 days, calculated features for users in each period and calculated target for each user on each corresponding 62-day window. Then those 4 dataframes were combined in one train set.

**Feature engineering**
Very simple: 
- label-encoded categorical features, 
- some time features in 168-day window such as intervals between user’s first session and window’s start and from user’s last session to window’s end,
- some statistical features based on user’s pageviews and hits.
Features were calculated only on 168-day window data without using sessions from next 46-day window, because I didn’t use “external” data from demo account for 16/10/2018-30/11/2018 period.

**Problem as “classification and regression”**
Firstly, I predicted probability of returning of customer using all train set. Secondly, I predicted the amount of transactions for those customers who returned. So, in regression task train was filtered only by customers who had a session in  62-day window. As final prediction I multiplied probability of returning by predicted amount for returned customer.
Approach of direct prediction by only regression using all train set  gives only 0.88320 score (it’ll be the 6th place on LB). So the trick of adding classification component was crucial. 

You can find the code [here](https://www.kaggle.com/kostoglot/test-score#L144).
Note: Score is slightly different from the score on LB (0.88143 instead of 0.88140) due to different environment on my local machine.
