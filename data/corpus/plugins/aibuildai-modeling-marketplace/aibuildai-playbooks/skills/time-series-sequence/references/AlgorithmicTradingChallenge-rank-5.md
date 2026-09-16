# Winning Algo/Code

Competition: AlgorithmicTradingChallenge
Rank: #5
Source: https://www.kaggle.com/c/AlgorithmicTradingChallenge/discussion/1236#7812

<p>Thank you, CMCRC, for creating such an engaging and interesting competition.&nbsp; I would also like to thank all of the competitors for creating such a competitive atmosphere.&nbsp; Congratulations to Ildefons; you did an excellent job.<br>
<br>
I ended up not selecting my best score, but my private leaderboard score seems to have improved throughout the contest.&nbsp; I have just started learning statistics and R, and I picked up quite a lot throughout this competition.&nbsp; I hope to have the chance to compete
 against all of you in the future.<br>
<br>
The data that we were provided with for this competition was very interesting in many respects.&nbsp; I would like to discuss Christopher and Cole's observations before getting into some of my own points.<br>
<br>
1.&nbsp; I noticed that although most of the predictive value was concentrated after t=45, after correcting t=0 to t=45 for outliers(most notably errors right after market open, which I will get into later), there was some predictive value to be had in these values.&nbsp;
 Predictors based on data from t=45 to t=50 also had some unfortunate nonlinear tendencies, which were minimized when using longer time horizons.&nbsp; Observations from t=0 to t=45 were also very useful for creating predictors based on volatility.<br>
<br>
2.&nbsp; Much of the daily error was concentrated right at market open.&nbsp; There were secondary areas of high error and volatility at 10:30, 13:30, and 15:00.&nbsp; There was a theory posited on these forums that these secondary areas were a result of other markets opening,
 which created arbitrage opportunities.&nbsp; The arbitrage opportunities may have unfolded in a predictable fashion, but I did not have time to solve this issue.<br>
<br>
3.&nbsp; I also found that a &quot;per stock&quot; model underperformed a model trained on the entire data set.&nbsp; Models trained on specific subsets of time(because there was a definite time of day effect) also underperformed models trained on the entire set.<br>
<br>
4.&nbsp; I found that this competition, because of its design, came down to two distinct predictions problems.&nbsp; Competitors had to predict the bid-ask spread as it recovered from t=51 to t=100, but also the trend of bid and ask prices from t=51 to t=100.&nbsp; As one
 may expect, the trend was much harder to predict than the spread.&nbsp; The fact that the trend could be predicted at all is interesting, but the predictive abilities of the models I tried were not extremely strong, partially because it was difficult to test my
 trend models(testing on out of sample training data was problematic, because the training data has similar trend characteristics throughout, whereas the test data, sampled in a different way, had drastically different trends).&nbsp; The spread, by contrast, had
 much more uniform characteristics across both the training and the testing sets. &nbsp;<br>
<br>
5.&nbsp; There were several dozen rows in the training and testing sets that were possibly erroneous.&nbsp; These observations all occured at market open, and in them, the bid-ask spread was huge(up to 600 for one stock!).&nbsp; These large spreads affected linear models
 unless they were corrected.&nbsp; Their spreads from t=51 to t=100 did not recover in the same fashion as the typical spreads, and a significant portion of the error came from these rows. A full list of the rows that i suspect were erroneous in the testing set
 is attached to this post.&nbsp; Note that I used an automated methodology to select these rows, so not all of them may be erroneous.&nbsp; While there were approximately 200 of these rows in the testing set, there were only about 250 in the training set, which made
 them extremely hard to predict.&nbsp; The final outcome was doubtless influenced heavily by these rows, as alegro has pointed out.&nbsp; I noticed swings of up to .008 on the public and private leaderboards by varying the predictions on these rows alone.&nbsp; It appears
 that the 70% of the test set that was held out for the private leaderboard score contained significant outliers(as evinced by the higher private RMSE scores vs public, although I could be incorrect).&nbsp; Correcting for these outliers(or perhaps even one or two
 observations) was a primary goal for many competitors, I am sure.<br>
<br>
6.&nbsp; I agree that the different sampling methods for the training and the testing sets affected the outcome of the competition heavily. The testing set was biased towards observations from the beginning of the day.&nbsp; The training set contained a large proportion
 of data from the beginning of the day, but also had a lot of observations from the end of the day, leading to a somewhat U-shaped time vs amount of observations graph.&nbsp; Attempting to correct for these variations did not aid my model, however.<br>
<br>
7.&nbsp; The algorithm that was used by the CMCRC data provider to clean the data prior to it being delivered to us was also relevant to the competition.&nbsp; I used time series outlier filtration, which uses a process found in a few papers that involves moving averages
 and a distance of 3 standard deviations to filter large outlying values from tick data.&nbsp; I found very few observations outside of this threshold.&nbsp; Additionally, all the ticks were in order, and none, aside from a few at the beginning of the day, could be unequivocally
 called &quot;bad.&quot;&nbsp; Knowing how the data was removed and filtered prior to is being delivered to us might have impacted the accuracy of our predictors.<br>
<br>
8.&nbsp; Volume information would have been a huge boost to the predictive capability of my model, and I am sure those of many others.&nbsp; It would have aided in establishing how deep the limit book was at any given time.&nbsp; I would suggest that future competitions focused
 on predicting liquidity shocks address the outlier problem, shorten the prediction time horizon to 25, and give volume and timestamp information for every trade.<br>
<br>
9.&nbsp; The data exhibited significant heteroskedasticity, but I had little luck solving the issue with weighted models, clustering, or &quot;per-stock&quot; models.&nbsp; Feature selection helped to mitigate the issue by minimizing predictors that showed significant heteroskedasticity,
 but I was never able to solve the issue to my satisfaction.&nbsp; Did anyone manage to do so?</p>
<p>Again, thank you to CMCRC and all the competitors for creating such an engaging opportunity.&nbsp; I am very interested in finance and algorithmic trading, although my background is not necessarily in the field, and this was a good way to model tick data, which
 is usually very hard to obtain.</p>
