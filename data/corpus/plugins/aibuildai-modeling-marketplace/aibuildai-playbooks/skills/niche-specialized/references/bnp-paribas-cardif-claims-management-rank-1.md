# #1 Dexter's Lab winning solution

Competition: bnp-paribas-cardif-claims-management
Rank: #1
Source: https://www.kaggle.com/c/bnp-paribas-cardif-claims-management/discussion/20247

So glad we won this one! After having such good data findings it would have been a shame not to win :)

It is clear that this competition was not about ensembling but feature engineering, and we are happy that we found out the major key to winning this competition early on.
Therefore, we had lots of time to exploit our findings to the max and maintain the top position ever since. 
The trick was very simple, and I even shared a hint to that early in the competition (without me understanding its importance at that time):
https://www.kaggle.com/c/bnp-paribas-cardif-claims-management/forums/t/19240/analysis-of-duplicate-variables-correlated-variables-large-post/110095#post110095

We figured, that data unanonymizing was important and in the end we were able to figure out some of variables' meaning:

v40 - date of observation

v40-v50 - contract start date

v50 - days since contract startdate (in other words, claimdate)

v10 - contract term in months

and few others for remaining days of the contract:

v12=v10*(365.25)/12-v50 

v34=v10*(365.25)/12-v40

Now add this knowledge to the assumption that v22 is a customer and v56/v113 is a product type, 
and you will see very obvious patterns, that one contract may have several claims during contract lifespan (R: group_by(v22,v40-v50) %>% arrange(v50)):

[v22,v40-v50]   [v50] [target]

[ZLS 12840]   197      1

[ZLS 12840]   962      1

[ZLS 12840]  1437      1

[ZLS 12840]  1498     NA

[ZLS 12840]  1501      1

[ZLS 12840]  1726      1

[ZLS 12840]  1788     NA

[ZLS 12840]  1882     NA

[ZLS 12840]  2418      1

[ZLS 12840]  3352     NA

[ZLS 12840]  3370     NA

[ZLS 12840]  3909     NA

So we could drop the categorical {v22} level i.i.d. assumption, and move to panel data structured categorical levels such as: 
{v22,v40-v50} => sort(v50), {v22,v40-v50,v56} => sort(v50), {v22} => sort(v40), etc; 

Just looking at the data this way, it was obvious that target value is very persistent for each {v22,v40-v50} level - 
i.e. if a time series starts with target=1, it usually ends with target=1; if it starts with target=0 it often ends with target=0 too; claims which target shifts 0->1 are quite rare (1->0 only few cases);
So in the end it all resulted to correctly imputing target sequence for each level - which we done using lag(target) and lead(target) variables;
To our surprise, these variables were not overfitting LB, and in the end we made so many lag/lead variables that it was possible to drop v22 column, 
and didn't even use v22 mean target techniques disccussed in the forums, which probably many top teams did anyway.

Our best single xgboost model achieved 0.42347 public LB (0.42193 private), and the model takes itself only about half an hour to train on 12-thread cpu. 
To seal the deal, in last 2 weeks we experimented with other techniques and build few stack models with tens of different methodology models, such as nnets, 
linear SVM's, elastic nets, xgboosts with count:poisson, rank:pairwise, etc.
I personally enjoyed working with regularized greedy forests, which were almost on par with xgboost.

The role of ensembling may not as important as other competition though we have tried several diverse models. 

-models with tsne feature from continous variables

-models by levels of certain varaible (for example,var5) 

-knn models on likehihood encoding varibles



To sum up: 
As most of you, we were stuck at 0.45x for a long time and it took 3-4 weeks of dedicated time of looking and exploring the data in Excel to end up with panel time-series data, which was the key to success.
Having such knowledge about the data could have gotten you to top10 without too much effort.
In the end we created a useless model for Bnp, as our lead(target) variables use information from the future:)

And for guys who wants to succeed - 

a) when starting a new competition, create simple xgboost model and use feature.importance to get a nice start to discovering important features - then stop making models and work with the data.

b) do not underestimate the power of knowing what data you are working with

c) dedicate some time for data exploration and try to understand how people visualize the data in the forums

d) look for data patterns, especially if it has many categorical variables

e) spend some time reading forums of past competitions, especially winning materials

f) keep eyes on overfitting


P.S. special thanks to this blog post http://blog.kaggle.com/2016/03/17/airbnb-new-user-bookings-winners-interview-2nd-place-keiichi-kuroyanagi-keiku/ 
"I found that the out-of-fold CV predictions of categorized lag features were very important. As far as I saw in the forum, many of the participants may have not created these features."
This shook our beliefs and assumptions about data being i.i.d in {v22} levels, and it took us only 1 day to utilize this and claim the top1 rank for the rest of the copmetition:)

P.S.S huge thanks to Laurae for the input to data exploration. I believe it made this competition more interesting for most of us.

P.S.S. team name came up when I decided to join forces with Davut, and the cartoon was playing in the background- for both of us it was our one of favorite childhood cartoons :)
