# 3rd place solution

Competition: home-credit-default-risk
Rank: #3
Source: https://www.kaggle.com/c/home-credit-default-risk/discussion/64596

Congratulations to teams in first 2 places and to everyone else who took part in this competition and learned something new!

This was the biggest competition by number of teams participating so far and definately was not an easy one. 
I think it was also very interesting competition with rich dataset which gave wonderful possibilities for improving ML and feature engineering skills - thanks to Kaggle and Home Credit for this!

We didn't use any leaks, duplicate rows or other kind of post-processing tricks. Only pure feature engineering and ML models.

Next we'll give some short description of our individual approaches.

Evgeny approach
---------------

I built model based on my previous experience in credit scoring (last year I worked  8 months in a bank). Features were created according to my understanding of scoring process and the data. 
I tried to create a model not very far from a real process. It was interesting how far I could reach with this approach against Kaggle without stacking  As I had limited time, my model not perfect but good for further improvements - 6th place now alone. 

My main model was a mix of features and metafeatures. I built separate models for each block of data - base application, last application, bureau, credit cards, installment. I used  predicts and also some features from sub models (or created for those models but not used) in the main models.
At the end my main model had 124 features. Together in all models there were near 250 features from more than 1000 created.

I selected features by local cross validation one by one. My models were compact and each check iteration was very fast. During one night it was possible to check which features could be added or deleted with cv improvement. This way helped me to control my model better. Near the end of the competition I had a divergence between CV and LB and tried to find better subset for validation, but without success and return to full train.

Models for bureau and prev application were build unusual way - I used data without aggregation by sk_id_curr and used target from main application (same for all data for one client). I was very weak model Predicts were averaged by sk_id_curr after. This approach performed better in main model compared to model with aggregated data with better individual score.

I also built models to predict ext_sources features. I use main model for that. Predicts and differences with original were useful features. I also tried to built model to predict income, but without success.

First month I developed model locally without submissions and I missed gap between CV and LB. When I submitted first time, I expected some gap as many reported, but LB score was in line with CV (hear .800). It was little bit disappointed but I trusted my CV. Before merge I reached .804 CV &amp; .805 LB and still not realized what I did wrong as I know that my model far better than public.

I offered to alijs try to reach the prize together as he had model based on public scripts but much better. We had good effect from join.

After merge I took some top features from alijs models and he took some mine and our models became much better.

alijs approach
--------------

I started this competition quite long ago and got quite high (about 10th if I remember correctly), but switched to Santander competition at some point and was slowly dropping down the LB here.
When Evgeny offered to create a team about a week before deadline, I instantly agreed, as we had already worked together in one team previously with very good success.

As I don't have any experience with credit scoring, my features were mostly created from statistical point of view (different aggregates, etc). 

There were a lot of teams in top of LB with many members so they probably had a lot of diverse models to choose from. 
As we were only 2 members in the team, we had to find alternative ways to get as many diverse models as possible to be competitive. 

So for example I took about 50 different model runs of my experiments regardless of their CV scores, selected 7 most uncorrelated ones and made average of them - got quite solid CV score literally out of nothing.
My take-away from this - don't throw out anything until the competition ends, maybe it will be useful at some point later.

For combining our very different models together I made 4 second-level stackers (actually more, but others were dropped) and our best submission (Private LB: 0.80511, Public LB: 0.80969, CV: 0.80855) was simple average of those 4:

- LightGBM (CV: 0.808419)
- Random Forest (CV: 0.807978)
- Extra Trees (CV: 0.807539)
- Linear Regression (CV: 0.807291) 

Each stacker was trained on slightly different subset of about 15 first-level model predictions (mainly LightGBMs, but also some XGBoost, NN) and some selected raw features.

For the final submissions we selected one submission with highest LB score and the one with highest CV. The latter turned out also to be the one with highest private LB score. That just confirms one more time - trust your CV.

Good luck!
