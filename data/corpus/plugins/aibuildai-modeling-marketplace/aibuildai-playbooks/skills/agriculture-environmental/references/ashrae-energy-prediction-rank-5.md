# 5th Place Solution

Competition: ashrae-energy-prediction
Rank: #5
Source: https://www.kaggle.com/c/ashrae-energy-prediction/discussion/127086

First of all, we thank the kaggle and ASHRAE teams for holding the competition!

I'm sorry my post was delayed due to my laziness.

Our solution is not sophisticated and I think there is much room for improvement.
(I want to apologize in advance, I don't use English in everyday life,
so I'm not good at English)


Below is an overview of our solution.


## Pre-proceeding
We dropped rows such as
* Long streaks of constant values
* Zero target values (only electricity)  

By removing these data, the score was greatly improved.

## Feature Engineering
We try two kinds of target encoding

### 1. percentile for each building_id, meter
As shown in the figure, the 5th and 95th percentile of the target value was calculated for each building_id and meter, and we used these features.


In our case, these features improved the score.

### 2. propotion


For each building_id, we apply these process.
* Calculate median of target value per day of week.
* Calculate its proportion (see figure).

This is an example of day of week. We also apply this technique to hour, day, and so on.


## Modeling
* Using only LightGBM(train for each meter)
We apply two-step modeling
### Step1: Determine num_boost_round for each building_id 


* Define training data(2016/01/15 ~ 2016/05/31) and validation data(2016/09/01 ~ 2016/12/31)
* Training with LightGBM and find early stopping round for each building_id(n1 ~ n1448).

### Step2: Train with all train (year 2016) data and predict test data


Training with all train data and predict test data.
The number of trees used for prediction were changed for each building_id (using n1~n1448 obtained in step1).

This approach improved the public score, but the private score did not improve much.

## Ensemble
* Used leaked data(site 0,1,2,4,15).
* Weighted average for each meter and year(2017,2018).

We also used other competitor's submission files.
sub1: https://www.kaggle.com/purist1024/ashrae-simple-data-cleanup-lb-1-08-no-leaks
sub2: https://www.kaggle.com/rohanrao/ashrae-half-and-half

## Submission
After ensemble, we achieve 1.047 on public LB) / 1.236 on private LB 
(1.058 on public LB / 1.272 on private LB in our single model)

<br>
If you have any questions, feel free to ask.
Thank you for reading.
