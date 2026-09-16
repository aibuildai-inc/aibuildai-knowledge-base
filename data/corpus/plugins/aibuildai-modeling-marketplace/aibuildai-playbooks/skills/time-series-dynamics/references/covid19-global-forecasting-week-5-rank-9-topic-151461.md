# Linear Regression Is All you Need (9th solution week 4)

Competition: covid19-global-forecasting-week-5
Rank: #9
Source: https://www.kaggle.com/c/covid19-global-forecasting-week-5/discussion/151461

I said I'll describe my solution if it ended up in top 10, so here it is.  

# Acknowledgements

First of all, I'd like to thank Kaggle for this challenging series of forecasting competitions.  I say challenging for two reasons.  First, this is about people being sick and people dying.  It is hard to abstract from that dire reality.  Second, one week sprints are too short for me.  I need more time to immerse in a given machine learning problem.  In this case it took me 4 weeks and 2 days to find a model I was happy with.

I also want to thanks my teams mates for first 3 weeks, Ahmet and Giba.  They preferred to go solo last week but I would have been happy to stick as a team.  We used very different models (NN for Ahmet, xgb/lgb for Giba, lr for me) with he hope that together it would work well.  It did actually in week2 but we didn't select it at the last minute because of some cv issue.  It would have landed us at rank 8th unless mistaken.  Week 3 was a bit disappointing for us.  

I also warmly thank those who curated the additional datasets I used, especially Vopani.  Using these made a difference.

Last but not least I want to congratulate those who did consistently well in these competitions, and especially those who did well in week 4.  @david1013 seems to have great models, even when not using an extra day of data if I get his recent posts correctly.  It would be interesting to know Lockdown and Kaz model score when not using an extra day as well.  Paulo Pinto performance in week 4 is awesome.  I can't name all but let me mentions PDD team who did well in several weeks too.  

# Modeling Goal

I decided early on to not approach this as I usually build forecasting models.  Reason is we are modeling a physical phenomenon where people get infected, then infect others for some time, then either recover or die.  I therefore studied [compartmental models used in epidemiology](https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology), SIR and the like.  These models rely on two time series: cases and recoveries/deaths.  If we have accurate values for both then we can fit these models and get reasonably accurate predictions.  

Issue is we don't have these series.  

For cases we have a proxy, confirmed cases.  This is a proxy in many ways:
-  It depends on the testing policy of each geography.  Some test a lot, and confirmed cases are close to all cases.
- A large fraction of sick people are asymptomatic, hence are easily missed by testing.
- Testing does not happen when people get infected or contagious, it often happens with a delay.
For all these reasons the confirmed case nubers we get is a distorted view of actual cases.

For fatalities the numbers aren't accurate either;
- In some geos we only get deaths test at hospital, in other geos it includes fatalities from nursing homes.
- We don't have recoveries data.
The latter can be fixed by grabbing recovery data from other online source.  This has been done by some top competitors, I wish I had done it.

Despite all these caveat, I assumed that we still have some form of SIR model at play with the two series we have at hand: fatalities depend on cases detected some while ago.   That led to my first model.

# Fatalities Prediction Via Linear Regression

If we assume that fatalities are a proportion of cases from, say, 8 days ago, then the proportion can be computed from log scaled data.  See examples below for some countries.  Green curve is the difference between confirmed cases (blue) and fatalities 8 days later (orange)

Italy:




France:



Spain:


Then running a linear regression on log of past cases as input, and log of fatalities as output should capture this dependency.   I published a simplified version in [a notebook](https://www.kaggle.com/cpmpml/fatalities-prediction-via-linear-regression).  The above pictures are taken from it.

The fact that the proportion depends on the geography can be captured via additional input features.  In practice I found that log fatalities increase is a linear regression of cases from between 15 to 5 days ago.  These limits were tuned via cross validation.  They match what researchers say about Covid.

# Cross Validation

We used time based cross validation for our models.  We selected 3 to 5 folds, depending on the week competition.  Each fold uses data until a given data for training models, and data after for validation.  One fold was defined by the actual test data: validation starts at the first day of test data, even though train and test data overlap.

Models in our team were trained to predict next day.  To predict on a longer period we append the prediction to train data, then apply to model on it shifted by one day to get the prediction for the second day.  We repeat for all remaining days in validation or test.  Other teams trained one model per prediction day.  I didn't thought of it but it is definitely doable given how fast training my model is.

# Predicting cases

For cases I used an auto regressive model, i.e. linear regression using past values as input.  That model was not that bad, but it turned out to be pessimistic.  The flattening of curves we observed wasn't captured correctly.  I spent the last 2 weeks thinking about how to model this.  I tried to revisit the epidemiology models, but couldn't find a way to reliably fit them for all geos.  I didn't want to add an arbitrary decay, or fit a curve either.  I therefore turned back to data exploration.  And the day before deadline I found that a log of the decay in daily cases was very interesting, see some plots below.



It means that daily cases decrease linearly in the log space.  It means that the number of daily cases is of the form N - exp(a t) where N is a starting population, a a constant, and t the number of days.  I explain this derivation [here](https://www.kaggle.com/c/covid19-global-forecasting-week-5/discussion/143791).

The model is therefore fitting a linear model on the log of cases over time.  

This model works well in geos where the number of cases is high, but fails miserably in geos where epidemic starts.  I therefore blended it with the previous one, with weights depending on the number of cases a while ago, 12 days if I remember well.  This model improved CV scores tremendously compared to the previous one.  It was modeling curve flattening rather well, at last.

I then tried the same with fatalities:  They indeed decrease linearly in the log space, but the data is sparse: I could fit a linear regression reliably only for geos with a rather large number of cases.  Fortunately for people and unfortunately for my model, there were not that many geos like this.  Then I realized I could use the same trick as in my very first model: log of fatalities depend linearly on past log cases.  I could therefore fit a linear regression on past log cases as input, and log of fatalities as output.  I used the same past period, i.e. cases between 15 to 5 days ago more or less (I don't remember the exact figures).  My fatalities model is therefore a blend of the original linear model from past cases, a linear regression fit on past log fatalities, and a linear regression fit on past log cases.  

That's what I submitted.  

# Post Deadline

I am rather pleased to see that a simple blend of linear models was competitive.  I wasn't sure I could handle differences in geos via adaptive weights when blending, but it seems it was good enough.

After one day of rest after deadline I realized I could have done a better job.  If fatalities depend on the slope of the log of cases, and if that slope is constant, then why estimate it on days from 15 to 5 days before, and not on days from, say 10 days to current day?   Using more recent data is always better, then why not here?  I implemented this small change and used late submission.  This improved model score was similar to the one I submitted in time until last week where it started to outperform it.  Its final private Lb score is 0.41840.  Bottom line is it took me 4 weeks and 2 days to get the right model...

# Week 5

I contemplated entering week 5 but didn't for 3 reasons.  One reason is I was bothered by the extra day data from world odometer.  I saw people who used it in week 4 ask for it to be forbidden.  It would have been a smart move IMHO.  Another reason, more important, is that a one week sprint with changes in competition setting was too short for me.  Last reason may be even more important.  I think my model works better when there are high level of cases and fatalities.  The inclusion of smaller grained geos like US counties probably mean that my model would not be a great fit.

I admire those who entered week 5 and I'm looking forward to their writeups when the competition completes!
