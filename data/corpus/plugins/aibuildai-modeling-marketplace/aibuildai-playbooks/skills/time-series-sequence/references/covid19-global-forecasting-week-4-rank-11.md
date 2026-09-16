# PDD Writeup

Competition: covid19-global-forecasting-week-4
Rank: #11
Source: https://www.kaggle.com/c/covid19-global-forecasting-week-4/discussion/146754

Thanks to my teammates @dott1718 and @christofhenkel for collaborating throughout the four competitions. As even the last week competition is already close to half-time in terms of scoring, we want to give a short overview over out solutions. We are really happy with our results across the four weeks, specifically as more optimistic models (predicting fewer cases and fatalities) performed better and our most pessimistic models in week 2 performed the worst.

**Week 1**

Everyone went into this competition with a lot of fear and uncertainty about covid-19 and I think this also reflects in the models that were produced. Very little data was available and most countries were in earlier periods of severe exponential growth. We quite quickly figured that utilizing any ML model does not seem appropriate here due to lack of historical data. We thus crafted the basis for our main model that we utilized throughout all four weeks. For predicting confirmed cases, the model uses a constant growth rate that is reduced by a decay that is accelerating over time. For fatalities, we multiply predicted cases of a given time lag with a fatality rate for a given country, and also add a constant growth.

This model allowed us some flexibility also in choice of (hyper-)parameters. We ended up deciding on one optimistic and one pessimistic submission in week 1, where the optimistic one has lower growth rates and higher acceleration of the decay of this growth rate and the pessimistic one has it the other way around. For countries where we already have "sufficient" data above a certain threshold, we learn the growth over a certain time period, and for others we employ default growth rates.

Optimistic sub: https://www.kaggle.com/dott1718/cv19-by-growth-rate-v5-09?scriptVersionId=30833479
Pessimistic sub: https://www.kaggle.com/dott1718/cv19-by-growth-rate-v5-03-per-5?scriptVersionId=30833633

The optimistic sub predicted globally 2.6 mio cases after 30 days and 120k fatalities after 30 days as of 22nd of April. The pessimistic one was way more doomsday and predicted 24 mio cases and 430k fatalities. The optimistic one ended 2nd on private LB and scored 1.07716 while the pessimistic one scored still 1.29940 and would have been 5th rank. This also gives you a good idea of the problems of this metric, because even if you are off that far globally from the actual numbers, if you get some low-case countries right, the metric improves drastically.

Now looking back, I have to say that our global estimation here was really, really precise. I just checked worldometer data, and on 23nd of April the actual confirmed cases were **2,722,000** and what we predicted was **2,650,252**. I would have not thought that we can be so precise there, even if individual countries can be off. 

**Week 2**

Even more exponential short-term growth was observed in the pre-ceding week. We mostly re-utilized our models from week 1 and adapted them a bit and overfitted them thankfully on short-term growth patterns which made them quite pessimistic in retrospective. The second submissions blends a power law fit with exponential cutoff to the model and it is actually our better sub here. 

Sub 1: https://www.kaggle.com/dott1718/cv19w2-2-sub?scriptVersionId=31270684
Sub 2: https://www.kaggle.com/philippsinger/cv19w2-pl4-sub?scriptVersionId=31271930

**Week 3**

We saw that machine learning solutions slowly start to shine, specifically for short-term predictions and decided to one the one hand further tune our model and sub it, and as a second submission blend some of the public ML solutions in. In detail, we used subs from @cpmpml (+team), @gaborfodor and @osciiart - so thanks to them! We heavily overweighted our model and the one from oscii in the final blend, so the weights were 8-8-1-1.

Single model sub: https://www.kaggle.com/philippsinger/cv19w3-2-v2-play-2-v3fix-sub-last6dayopt?scriptVersionId=31654736
Blend sub: https://www.kaggle.com/christofhenkel/cv19w3-v3fix-cpmp-oscii-belug-full-8118?scriptVersionId=31655753

Our blend sub predicts 3,668,603 confirmed cases for 7th of May and 295,911 fatalities which is ~10 days in the future. I think globally this solution will also come quite close to reality. We are also currently ranked 3rd in the competition with trending upwards, even though the gap to 2nd might be too large.

**Week 4**

Again, we decided to improve our model and blend it with other solutions. The big change we did to our model was that we do not rely for fatality predictions anylonger on predictions on confirmed cases but rather also take the individual fatality curves for future predictions. We also blend our model now with solutions from @osciiart, @rohanrao and @david1013. Our two subs this time have different weights for the blends, usually overweighting our model. We also fully rely on our model predictions for Chinese countries.

Sub 1: https://www.kaggle.com/christofhenkel/cv19w4-full-pdd-oscii-david-vopa-sub1?scriptVersionId=32072839
Sub 2: https://www.kaggle.com/christofhenkel/cv19w4-full-pdd-oscii-david-vopa-sub2?scriptVersionId=32072928

Sub 1 predicts globally on 14th of May 3,898,447 cases and 278,848 fatalities while sub 2 predicts 3,802,472 and 224,623 respectively. In general it seems that public subs tend to agree way more on cases and fatalities seems to drive most of the score differences. We are currently positioned 5th, but trending down a bit. Will be interesting to see how scores evolve.

**Learnings**

I am writing this write-up while hearing the news that Austria (the country I live in) removes many of the restrictions in place from May 1st onward. We have had one of the best curve flattenings globally, even though having huge exponential growth in the beginning and we quite overpredicted Austria from the start thankfully. The country implemented strong restrictions very early on and I believe they have helped a ton. This is something that we can learn from the most I feel like. Active cases (from worldemeters) looks like that for Austria:



I really hope that this trend will not reverse again, many other countries also look that promising and we can only hope for a bright future.

Predicting these pandemics is very hard, mostly due to lack of data, very large uncertainty, difference in reporting, differences in public behavior, and so forth. Still, I believe the models produced in these competitions can be helpful in the future, even if it is only due to our better understanding how hard it is to predict it. A lot of personal judgement flows into these models and predictions, making them also very subjective and thus combining multiple models and opinions seems to be also worth-wile to follow up on in future.
