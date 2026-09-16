# #1 solution - generalization with linear regression

Competition: godaddy-microbusiness-density-forecasting
Rank: #1
Source: https://www.kaggle.com/c/godaddy-microbusiness-density-forecasting/discussion/395131

New:  I've uploaded the formal model submit file that was required to receive the prize distribution: https://www.kaggle.com/competitions/godaddy-microbusiness-density-forecasting/discussion/425000

Code here - https://www.kaggle.com/kaggleqrdl/first-place-code, see below for an explanation.

**Acknowledgements**

Let me first acknowledge Kaggle, the Venture Forward folks, and GoDaddy.   Not just for creating Kaggle itself, hosting the contest and providing the microbiz data - incredible feats of faith, truly -  but also for being patient with my fixation on transparency.  I am an egalitarian thru and thru, and firmly believe transparency is what gets us there.

**Introduction**

The contest was to predict microbusiness density 3,4,5 months lookahead for mar/apr/may 2023 per each of the 3135 US counties in the dataset.  Microbusiness density here is roughly equal to the # of GoDaddy [internet domains registered](https://www.kaggle.com/competitions/godaddy-microbusiness-density-forecasting/discussion/374293) / adult population in a county as measured by a 2 year lagging US census.  

Eg: if a total of 10 GoDaddy domains were registered in [De Kalb, MO](https://en.wikipedia.org/wiki/De_Kalb,_Missouri)  with 200 adult population according to the 2021 census in 2023 February, than microbusiness density for that county would increase by 10/200 or 0.05 in that month.

We were given historical microbiz data for 2019 August -> 2022 December to train with, and a not-so-secret 'public LB' of data for January to validate against.  Scores on the public LB were provided by Kaggle, and we were given 5 submissions per day.

**Method**
To win this, I took a baseline prediction for the january public leaderboard based on overfitting it, and used linear regression to project into mar/apr/may

```python
'sklearn.linear_model.LinearRegression',  ['pct_college_2021', 'l11_uemp', 'shiftf1_l1_active',]  - 4 month lookahead
'sklearn.linear_model.LinearRegression',  ['shiftf1_l1_active_logit', 'pct_college_2019', 'l4_active','l10_lf'] - 3 month lookahead
'sklearn.linear_model.LinearRegression',  ['l8_uemp', 'l1_ur', 'l11_uemp', 'l10_ur', 'pct_bb_2017'] - 2 month lookahead
```

1. l# here means lag # of months, shift is the actual value, just l#_ is pct_change
2. shiftf1_l1_active_logit is whether there was a change in active over previous month (l1_active != 1)
2. pct_college_2021 / pct_bb_2017 is from the census data the hosters shared
3. uemp / lf / ur are from the dataset I shared here - https://www.kaggle.com/datasets/kaggleqrdl/gd2022datasets

I selected these features by cross validating on the last possible window (2,3,4 month respectively) without overlap.  

**Insights**
The key ideas that seemed to help:
- LB overfitting / probing.  I did this by using the top public LB notebook plus focusing on the 10 or so counties that had recent significant volatility in the dataset.  This was done because in non stationary time series data, the last value is your most powerful and compelling signal.  It's worth noting that due to the [dec22/jan23](https://www.kaggle.com/competitions/godaddy-microbusiness-density-forecasting/discussion/413974) snafu (we were given 60% incorrect data, as measured by SMAPE), this technique was severely impaired and my lead in the public LB went from ~0.2 to ~0.08 over the top 20.  Overfitting the LB doesn't work very well when the LB has wrong data.   
More explanation here on this approach - https://www.kaggle.com/competitions/godaddy-microbusiness-density-forecasting/discussion/413109  I also explained these ideas early on in the contest [here](https://www.kaggle.com/competitions/godaddy-microbusiness-density-forecasting/discussion/375244) and [here](https://www.kaggle.com/competitions/godaddy-microbusiness-density-forecasting/discussion/378002). The [notebook](https://www.kaggle.com/kaggleqrdl/first-place-code) has code used for SMAPE root solving and probing techniques
- LinearRegression - I tried practically every(*) other model and techniques, even dl/nn.  LR consistently resulted in more reliable CV scores.  Overfitting is always a concern with time series data and it's so easy to tune other models and convince yourself you found something superior.
- Early stopping on greedy feature selection.   This was done mostly on an intuition basis, but after running 1000s of experiments I developed a good feel for this I believe.  I noticed overfitting occurring when I did more than 4 or 5 features, plus the decreased rate of improvement was another signal that overfitting had occurred.  Certain features as well seemed to have more signal than others (eg, L10/11 employment, bb census) so I would keep going until they made it into the list but not much more.
- Last window CV for feature selection.  My hypothesis is that as a non stationary time series the relevant predictive features change and it's best to use the latest CV window (last few months of 2022).   This played out in the results I saw over the experiments I did, but I will say std deviation on the error term was quite high, even if the mean was superior to longer CV periods.  The risk was that so little CV time made it a bit of a gamble.
- Use as much original training data as possible - Generalizing through data sufficiency was a reoccurring theme in this comp, and any technique that didn't appreciate it was degraded respectively.  I used everything up to the lag window as per above.  I didn't modify any of the data (beyond removing 3 problematic counties), even left the break in 2021 alone.  I tried doing all sorts of things to clean up that data, but they just seemed to degrade results.    
- rounding based on active/population of course was a must.  I found clipping, in the end, didn't help.

*every regression model in sklearn, various arima libraries (a lot of time here, wasted I feel), hierarchical, xgb, catboost, lgbm, pytorch/dll.  I tried both manually tuning the previous as well as various auto approaches like grid search/optuna.  Lots of very exciting results, but eventually it occurred to me I was probably just overfitting.

It's worth observing that my lead against the #2 and #3 spots increased significantly from the public LB versus the private LB.  I believe this was due to superior predictions for apr/may as my march results were somewhat lackluster.   My lead over the top 20 spots also increased, though it is less clear how much of the public LB influenced their models.

**Ideas that didn't seem as important**

I also tried a lot of different datasets (including all the ones I shared publicly) up to and including google keyword trends for 'godaddy'.  :)  The above worked best when combined with early stopping.  There would be sparks of good CV scores when using other features, but adding them frequently felt like overfitting.  

I briefly looked into using zone files and whois databases, but these were not openly available so of course I didn't use them. 

Crawling/Scraping would have in theory been public data, but I had committed to myself not to use any data that I didn't share openly, so I figured it wouldn't be worth the effort required. 

Everything I used I shared in the [dataset]( https://www.kaggle.com/datasets/kaggleqrdl/gd2022datasets) I made publically available at the beginning of the comp.

I did a lot of initial work around per county models, work I eventually decided was wasted.  There was barely enough data to validate the models and features above.  There might be a way of segregate the counties, the question is - Would you have to segregate your training data as well? Would the increased complexity be worth what might potentially be only incremental improvements?   Hard to say, especially since observing incremental improvements is mostly impossible when your CV scores are so volatile.  


**Future directions**
One thing I spent time looking at was correlation between the counties.  There is an argument to be made I think that different counties may follow similar trends at potential lag times.  There might be more to do here, but it would need to be done globally rather than individually, and your models would require appropriate data sufficiency. 

Another feature I didn't plumb carefully enough was 1/2/3/etc annual renewal on GoDaddy domains, an idea I got by chatting with some domain resellers recently on a reseller domain community forum.  I suspect they would provide a well of ideas around this that could be leveraged more.

**Final thoughts**

As I mentioned above, and others have mentioned, the data was non stationary except for some global linear growth coefficients.  It's also been said that the smaller counties added a lot of noise, but I believe it was the right choice to try to see if we could do something there.  There is a paucity of stats and analysis around more rural areas and emergent issues are hard to detect because of it.

Another way of looking at the data might be volatility instead of direction.  The wild fluctuations in active domains could just be due to data errors but they could also be due to underlying social and economic factors. In the latter case, we need to consider: could the volatility be an opportunity, like the first sparks of a fire? Could these counties benefit from more targeted and opportune investment? If volatile counties have increased potential, what are the triggers for volatility?  These are questions and experiments (signal + intervention) worth considering

If the VF folks would like to contact me (heh), feel free to reach out.  As you may have noticed, I could talk about this stuff all day long with folks who are equally interested.

..

There was a bug in the team update that didn't let me change the LB pointer which I had originally pointed to the message below, so you're stuck with both.  I originally pointed it here as I was more interested in what other folks did as I wasn't really that optimistic about my simplistic approach above.  In retrospect, I probably should have trusted the process of elimination that I went through.


previous thread title: **What is your mean active change over december for mar/apr/may?**

Edit - to make clear up front, you can share your results from your submission by just executing this code.  It uses microbusiness density and not active, but that should be fine for comparison.

```python
dfc = pd.read_csv("submission.csv")
rt = pd.read_csv("revealed_test.csv").set_index("row_id")
dfc['first_day_of_month'] = dfc['row_id'].apply(lambda x:x[-10:])
dfc['cfips'] = dfc['row_id'].apply(lambda x:int(x[:x.index("_")]))
dfc = dfc.reset_index().set_index("row_id")
dfc['mbd_chg'] = dfc.apply(lambda r:(r['microbusiness_density'] - rt.loc[f"{r['cfips']}_2022-12-01", 'microbusiness_density'])/rt.loc[f"{r['cfips']}_2022-12-01", 'microbusiness_density'], axis = 1)
display(dfc.groupby("first_day_of_month").mean()['mbd_chg'])
```

....

It's going to be awhile before we get final results, and I think May results might play a large role in determining the eventual winners.  While we wait, it might fun to share what we actually predicted.  We could compare our actual results, which would be good and I've uploaded my two [here](https://www.kaggle.com/datasets/kaggleqrdl/gd2022-subs) -  but for those who don't want to and perhaps simpler might just be sharing mean % change over active.

Here's the code I used.  It requires having accurate values for december in your dataframe and also have calculated 'active' as one of your columns.  

If you don't want to use active, you can also share microbusiness density as well, but I found it less interesting because of the pop shift.  There were some outliers in the pop shift that made it harder to ensure that nothing went awry, so I liked having active available.   I've shared both below.

You can calculate cfips/first day with this code:

```python
dfc['first_day_of_month'] = dfc['row_id'].apply(lambda x:x[-10:])
dfc['cfips'] = dfc['row_id'].apply(lambda x:int(x[:x.index("_")]))
```

Calculate active_chg values with this code:

```python
dfc = dfc.reset_index().set_index("row_id")
dfc['active_chg'] = dfc.apply(lambda r:(r['active'] - dfc.loc[f"{r['cfips']}_2022-12-01", 'active'])/dfc.loc[f"{r['cfips']}_2022-12-01", 'active'], axis = 1)
dfc.groupby("first_day_of_month").mean()['active_chg']
```

This is my seasonal model 

```python
first_day_of_month
2022-11-01   -0.00635502 
2022-12-01    0.00000000
2023-01-01    0.00328686
2023-02-01    0.00328686
2023-03-01    0.01505438
2023-04-01    0.02265933
2023-05-01    0.02476927
Name: active_chg, dtype: float64
```

This is my long trend
```python
first_day_of_month
2022-11-01   -0.00635502
2022-12-01    0.00000000
2023-01-01    0.00328686
2023-02-01    0.00328686
2023-03-01    0.00881651
2023-04-01    0.01215006
2023-05-01    0.01580625
Name: active_chg, dtype: float64
```

Using mbd code if you didn't have an active column:

```python
dfc = dfc.reset_index().set_index("row_id")
dfc['mbd_chg'] = dfc.apply(lambda r:(r['microbusiness_density'] - dfc.loc[f"{r['cfips']}_2022-12-01", 'microbusiness_density'])/dfc.loc[f"{r['cfips']}_2022-12-01", 'microbusiness_density'], axis = 1)
dfc.groupby("first_day_of_month").mean()['mbd_chg']
```

Seasonal sub
```python
first_day_of_month
2022-11-01   -0.00635502
2022-12-01    0.00000000
2023-01-01    0.01208731
2023-02-01    0.01208731
2023-03-01    0.02388208
2023-04-01    0.03161783
2023-05-01    0.03375395
Name: mbd_chg, dtype: float64
```

Long trend sub

```python
first_day_of_month
2022-11-01   -0.00635502
2022-12-01    0.00000000
2023-01-01    0.01208731
2023-02-01    0.01208731
2023-03-01    0.01764007
2023-04-01    0.02099663
2023-05-01    0.02467693
Name: mbd_chg, dtype: float64
```

Note for february I just used the january values. Sharing january isn't really that helpful of course, I just do it for fun here.
