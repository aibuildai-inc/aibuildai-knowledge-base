# [1st Place Solution] My Betting Strategy

Competition: home-credit-credit-risk-model-stability
Rank: #1
Source: https://www.kaggle.com/c/home-credit-credit-risk-model-stability/discussion/508337

##Foreword

Hello, this is yuuniee.
Thank you to the competition host Home Credit and staff, Kaggle staff and everyone involved.

The image below is a summary of my solution.

[image_]

The maximum Private LB Score for my model pool without Metric Hacking is close to about 0.53.

As everyone knows, this competition takes place in two phases.
Phase 1 was ML (Machine Learning), and Phase 2 was MH (Metric Hack).
I will briefly introduce each part.
(If you are only curious about ML, please read only Phase 1.)


## I. Phase 1 - ML (Machine Learning)

Thank you for writing a great notebook for EDA and solution construction at the beginning of the competition.
@sergiosaharovskiy - https://www.kaggle.com/code/sergiosaharovskiy/home-credit-crms-2024-eda-and-submission
@greysky - https://www.kaggle.com/code/greysky/home-credit-baseline
Additionally, I would like to thank the many participants who provided various insights.

1. The CV strategy is StratifiedGroupKFold, and has been tested and trained in various forms: No Shuffle and Shuffle. In my case, differences in CV of around 0.001 to 0.005 had a low correlation with LB, while differences above 0.01 had a high correlation with LB. In addition, CV improvement through model parameter adjustment had a relatively low correlation with LB, and CV improvement through FE had a relatively high correlation with LB.

2. There's nothing special about Feature Engineering. Max, Min, Avg, Var, First, Last and Max-Min Difference of each item were utilized. (I will release the code after cleaning it up.)

3. LGBM was a bit lacking compared to Catboost, but was good for the ensemble. The reason why the number of features is smaller is because some items (mainly categorical types) were excluded because they caused performance degradation and overfitting in LGBM, and other slightly different types of features were added and deleted.

4. DNN is a Denselight model and utilizes the LightAutoML library. There were some bugs, so I had to fix some of them myself, but overall, I think it is a good library. At first, I planned to lightly test with Denselight and then build the final model with a larger model (FT-Transformer, etc.). Surprisingly, I haven't created or found a model that beats Denselight performance. Perhaps I made a mistake, or the nature of the data is sensitive to overfitting, so I guess a simpler model worked better.

5. Catboost was the model that performed best in this competition and I guess it performed well on a large number of categorical features (about 117).

During this period, my public LB was close to top 5 and the competition seemed to be going very smoothly.
But...


>### + What didn't work
>1. Transformer type models such as Tabnet, TabTransformer, FT-Transformer, etc.
>2. Income, expenditure, and tax statistics by period (month, week, etc.).
>3. Differences between individual income, expenses, and taxes.
>4. K-means clustering


## II. Phase 2 - MH(Metric Hack)

Pointed out problems with MH at the beginning of the competition,
@at7459 - https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/discussion/476449
Announced the revival of MH in the latter half of the competition,
@johnpateha - https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/discussion/497167

Also, thanks to the many participants who suggested various other comments and solutions. Thanks to you, my knowledge has expanded.
In particular, the two people above are true scientists and journalists who could have hidden this discovery and profited from it, but pursued the public interest by exposing it to everyone. (Why not consider Kaggle giving a special award for this sharing?)
As a result, the problem was not solved, but we learned something(?) from it and at least I was able to take minimal measures.

After @johnpateha's "Metric hack again, sorry" post, I tried Data Reversing and found that the difference between date_decision and min_refreshdate_3813885D had a high correlation (about 0.9 or more) with WEEK_NUM.
(More information can be found at https://www.kaggle.com/code/eivolkova/how-to-restore-the-dates?scriptVersionId=180157891, published by @eivolkova immediately after the competition.)
And the way suggested by @at7459:

```python
DEVIDE = 1/2
REDUCE = 0.02
condition = df['WEEK_NUM'] < (df['WEEK_NUM'].max()-df['WEEK_NUM'].min())*DEVIDE+df['WEEK_NUM'].min()
df.loc[condition, 'score'] = (df.loc[condition, 'score'] - REDUCE).clip(0)
```

Based on this, the results of examining the score details submitted by changing the REDUCE and DEVIDE values ​​were as follows.

1. The public/private division of test data spans the entire period and it is difficult to identify a specific distribution, and the optimal value in private will probably be distributed around 1/4 to 3/4.
2. The optimal public LB value of REDUCE is 0.03 (at least in my model), and the optimal value in private is probably distributed around 0.02 to 0.04.

The difference in pure model performance between top competitors was 0.00X,
Since the difference due to the above adjustment value was 0.0X,
At this stage it became gambling for me.

Some will bet low, others will bet high.
Here, I trusted my model and chose neutral, and chose DEVIDE=1/2, REDUCE=0.03, which is the median of the expected distribution.
I expected someone who bet high or low to take the prize, and I expected about a 50% chance of finishing in the top 10.


## III. Conclusion

+ The first selected submission is the No Hack model with CV Best Score created in Phase 1, and the second selected submission is the application of Metric Hack to it. (I thought there was a high probability that Metric Hack would work, but I also submitted No Hack just in case)

+ Rather than simply sharing solutions, I thought it would be better to share my experience in detail. To improve the bad parts that happened in this competition.

+ My ranking was due to luck, and although it is the Best Private Score submitted, it probably won't be Best Solution.

+ Although the competition ended on a gloomy note due to the metric hacking issue, the host and staff worked hard to make it a meaningful and successful competition. 

+ Based on this, we expect a more developed system in the future.


thank you


>Edit : I posted the FE code here.
>644(for lgbm) - https://www.kaggle.com/code/yuuniekiri/fork-of-home-credit-risk-lightgbm
>661(for cat & dnn) - https://www.kaggle.com/code/yuuniekiri/fork-of-home-credit-catboost-inference
