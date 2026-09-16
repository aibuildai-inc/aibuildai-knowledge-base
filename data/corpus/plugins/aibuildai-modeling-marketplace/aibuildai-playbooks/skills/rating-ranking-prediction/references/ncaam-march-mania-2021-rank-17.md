# Congratulations to the top teams and summary of the experience - top3%

Competition: ncaam-march-mania-2021
Rank: #17
Source: https://www.kaggle.com/c/ncaam-march-mania-2021/discussion/230944

What a year!

This is my second time participating in this competition but the third time building the model. I am happy with the result because, due to a silly mistake on the day of the submission, it comes purely from the model results with no overwrites. We can call this the first happy accident since overly confident predictions would have torpedoed my score.

The model is my take on the classic @raddar model plus a logit and you find everything about it here https://github.com/lucabasa/kaggle_competitions/tree/master/march_madness

Both models trained on (different) sets of features coming purely from the provided data (this is more laziness on my side to use the advanced statistics other people use or the betting probabilities). You can find them all in this notebook https://www.kaggle.com/lucabasa/quick-eda-with-common-feature-engineering

The choice of the features was driven by 2 equally important factors:
* adding the feature should increase the score on at least 3 of the past 6 competitions
* the role of the feature should make sense. (For example, the number of assists was somewhat helping but also decreasing the probability of victory of a team, which did not make much sense to me so I left it out)

The XGBoost model also produced the predictions for the spread competition and I am happy to see it got in the 6th position there. 

The two models were in accordance 94% of the time despite being trained on different features and relying on very different approaches. I then combined the predictions of the two models by boosting the confidence over the outcome for those games where both models were pretty confident. I trusted XGBoost as it was consistently getting at the top50 of each of the past competitions, I could check that pretty quickly thanks to the work I have done in the past years that I briefly showed in this other notebook https://www.kaggle.com/lucabasa/march-madness-model-validation-strategies

Like many others, it failed in predicting the major upsets. The ORU run is not particularly concerning to me as the event seems to be rare enough, but I will need to give a good look at why my model disliked so much Oregon St.

Looking ahead, I think I used the same modeling approach for too many competitions, this year I was a little bit more clever in choosing the features but I feel there is not much more to learn from it anymore. Therefore, I hope to see this competition again next year and take some more risks by trying something entirely different. The core assumption that season performance can indeed predict the outcome of a single game, while reasonable, is not bringing better prediction than an educated guess and it is nice to think that we can do better than that.

I am very curious to see what the top teams did to get better predictions on those games and I take the opportunity to give my congratulations to all the gold medalists. It has been fun seeing the LB changing daily with all the upsets and, while we all benefit from some luck, I look forward to seeing the nice approaches that led to these results and learn from them.


p.s. this is also my first medal ever, which feels good, not gonna lie.
