# Some ML, A lot of judgement and luck

Competition: covid19-global-forecasting-week-4
Rank: #4
Source: https://www.kaggle.com/c/covid19-global-forecasting-week-4/discussion/143893

# Intro

The title may not seem very inspiring when you try to solve something so serious, but it is what worked semi-OK (for the first 2 weeks at least )  in this competition series( given a large horizon). I think predictions of that length (e.g 30 days ahead) require a lot of guessing and luck if you are not an expert epidemiologist (and quite possibly even if you are one too).

There are so many things which are unknown when you try to predict that far ahead, like:

Will the vaccine come out? (ok this one might not really be the case here - I think we are still early for that )  or an efficient drug?
Are the people really following the social distancing rules or are they going to?
what rules will the governments put in place?
How strict will the government be at enforcing them?
How will the infection rate differ based on how people socialise in different countries/areas? etc.

Needless to say that the data is not perfect.

 Short term predictions (maybe within 3-5 days or so) could be modelled reasonably well.  I hope you don't mind me [sharing a blog](https://www.h2o.ai/blog/modelling-currently-infected-cases-of-covid-19-using-h2o-driverless-ai/) that summarises some ideas on how this could be done for short horizons  on the same underlying data for active cases - many of the tricks mentioned there were applied here as well particularly in the **pre-processing of the data** and **post-processing** of the predictions . 

 I  still believe that an ML practitioner in tandem with an experienced epidemiologist as a combo can work better than any individual on her/his own, particularly if you try to make predictions at scale for many countries/counties/cities at the same time.  

The inference kernel (with commented code for training ) [for week 4 is here](https://www.kaggle.com/kazanova/script-with-commented-train-counts-for-zeros?scriptVersionId=32084418)

I have the same version with [training included here](https://www.kaggle.com/kazanova/fork-of-script-with-commented-train-counts-for-zer?scriptVersionId=32113592) :

(It scores a tiny bit differently 0.02549 versus 0.02551 than my uploaded models although I have used same seeds - maybe because I run on a windows laptop - or different lightgbm version - not sure) . 


# The ML part

The Power Growth model has worked quite well to map epidemics in the past and I think many people have made good implementations here. 

I have tried a simple implementation myself - the problem was that adjusting the growth was challenging without expert knowledge (at least for me)- so  so I tried to model growth specifically. Instead of predicting counts, or log counts, what I tried to predict was:

 " with what value do I need to multiply the previous cumulative value every day". For weeks 2,3, I modelled 60 separate models. One model for each plus+x day for the confirmed and fatalities. 

I used [lightgbm ](https://lightgbm.readthedocs.io/en/latest/)to build the models.

The features were : 
- In how many days ago was X cumulative count reached. Where X for confirmed cases was `[1,5,10,20,50,100,250,500,1000]` days and for fatalities `[1,2,5,10,20,50]`
- up to 10 lags of 3-size-window average rates for both confirmed and fatalities
- the actual confirmed and fatalities as row counts for day 0.

The target is the average-3-day rate for each one of the +x days. So instead of modelling the rate of `future1/current`, i took the average of  ` future1/current + future2/future1 + future3/future2` when predicting the target for day+1. That is because it has many spikes, so the average smoothness the target and (I think) controls for overfitting. 

For week 2, I only validated through observation - no cv. For week 3 I validated based on the week 2 results and for week 4 on both week2 and 3.  A **fundamental FLAW** of both week2 and week3 models is that they ignore the 0s. The model requires an initial value of &gt;=1 to add the growth , so I expect the models down the stretch to do much worse (unless I am lucky - both me but most importantly the world and we don't have new deaths). The rules I used to initialise values for the zero cases  were poor (to say the least) .  

In week 4, I have added a model that tries to predict the 0 cases modelling the difference in log counts between days .The model setup and features are the same, the only difference is that instead of only rates, the model also uses difference of log counts for the same windows. In that model I have also added USA county level data - i saw a tiny improvement from adding that (probably because it is heavily weighted towards 0 as the cumulative values are generally smaller).   Other than that, I have found no other external resource useful (for longer term predictions). I have tried adding geographical data and recoveries with no success. 

Sadly these models are not very powerful - but maybe not too bad either. I would have been much further down without **post processing** and I would be drifting down very heavily. 

The problem with this approach is that it completely ignores the metric and a small error in the rate prediction can make a series go berserk. I saw quite a few series that for example started with 20 confirmed cases and my model was predicting it will reach to more than 30k (which was obviously a bit too dangerous to accept). Another problem is that the models are heavily influenced by what happened in China - it is the area that has the most future points available (to compute the target). In China the treatment of the outbreak was handled very differently than other areas and generally the models expected a much quicker plateauing  (and different initial growth rates) than what is actually happening in many other parts of the world. This is where **post-processing** helped.

# Post Processing

**Before I elaborate on this - I heavily applaud and congratulate people that have modelled this without much judgement or post-processing. They have definitely added much more than this solution ever will**. I found not doing the post processing very challenging to get a highly competitive score (for the first 2 weeks at least). 


Back to the post-processing, this is the most important part of my solution. The ML model gave a curve to use as a basis and make decisions on it. The model was not too bad on cases with stable growth rates and counts of over 5,000 (approx). I created certain rules after observing most  of the series individually. **All the decisions I made on the post processing rules, were based on judgement** - nothing coded or something that could be reused as a logic safely without observing the series. To manage time, I have ignored cases with larger counts (which is very bad from an epidemiological point of view )  subject to RMSLE (that I generally don't like for this problem) .

The reasoning behind the rules: 

- "first_10_then_1_" : This rule was mainly used in China provinces for cases that still had some minor increments. it applies rule (e.g decay) in the first 10 days and then it stops the growth.  I anticipated that even if there are some new cases , China is now very efficient at containing the outbreak with the surveillance measures and the  health app system.

- "_last_12_linear_inter_" :  I am an optimistic person. if for some reason the model was not picking up that the rate at some point needs to start decreasing , I was forcing a linear decay in the last 12 days. I have no idea how this will play out. because we have not reached that point yet in any of the competitions. for example if the rate at day 19 is 1.12. In day 20 it will become 1.11 , then 1.10, etc. I do not have a great scientific explanation to tell you why I do this, other than I compared some of the curves against some others from the Chinese provinces at similar count levels and saw that 18 days was around the average the rates were still going up. Maybe not a great assumption given it was based on a geographical area only - but this is all I had at that point.

- " decay_x" : This rule is the most common one. I apply various decays in the predicted rates. A lot of it is based on expectations and comparing series with what happened to other countries/states when the counts were at similar levels , preferably close in geography. I also used my knowledge of what rules/measures have been applied to the countries when making these judgements . 

- "combination of all the above" : you can apply any decay on the first 2 rules

- "accelerate" :Being an optimistic person, generally, I have avoided using that. It is the opposite of decay. 

- "stay_same":   rate=1  - mostly applied to Chinese Provinces, low count countries, Diamond princess.

- "normal" : Don't change the rate - trust the model as is

- "fatality" : Rule applies only on fatalities - it excludes confirmed. For week 2 model, I completely ignored fatalities when made judgements. From week 3 onward, I started making special judgements for fatalities.

Now that I have a sample of making these decisions - when the competition is over, I am hopping to find a pattern that can be applied without judgement. E.g evaluate the judgements and find what would have been the correct decay. 

For the last week. I created a simple file that has the predictions of the new model without applying most of the rules versus the predictions of the model of the previous week and saw how over/under I was. I used this to aid my decisions. I have a screenshot attached.


# Luck/guessing

I have taken many guesses in this competition. For example:

- China won't generally have new fatalities or many new confirmed cases especially in the smaller provinces. Same with Diamond Princess. 
- Tibet will remain with that single 1 confirmed case . If this goes to 1000  -, maybe  I am out of the competition.
- For 2,3 weeks that generally 0 fatalities -will remain 0 unless confirmed cases exceed 400. 
- News and Dashboards for recent information and trends (from what people have posted on the forums)
- Countries that enter later are more prepared/cautious about the virus and are better equipped to deal with this, since they have an idea for what measures might be more effective and/or how serious this can be. I have applied stronger decays to countries that enter now. 


Happy to answer any questions.
