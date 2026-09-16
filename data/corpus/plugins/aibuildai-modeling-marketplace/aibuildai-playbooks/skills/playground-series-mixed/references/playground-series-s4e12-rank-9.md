# 9th place solution: With a little help from my (Kaggle) friends

Competition: playground-series-s4e12
Rank: #9
Source: https://www.kaggle.com/c/playground-series-s4e12/discussion/554377

The month of December got off to a terrible start, as my father passed away. He'd been battling metastatic prostate cancer, so it wasn't unexpected, but it still was (and is) terribly hard. One of the many consequences was that I couldn't/didn't participate properly in this month's TPS episode for the first two weeks. Another side-effect was that my Kaggle login streak ended at 285 days - I'd been looking forward to take it to a year, but don't feel particularly motivated to start over. I'll just login regularly, and forget about such trivialities. 
Anyway, so by the time I started to truly get into the swing of things with regards to this episode, there were already several great notebooks to peruse, notably those of @martynovandrey & @backpaker, which were highlighted by @cdeotte in a post. @martynovandrey's excellent notebook in particular provided the first fully reproducible results under 1.40 (even 1.30, for that matter) & as far as I can tell, remained the best scoring fully reproducible notebook (that wasn't just a blend of other submissions) until the very end. I edited these notebooks to save the OOFs, and repeated the same with a few other interesting notebooks. I also borrowed features from various sources - @cdeotte's insightful post about deriving information from NANs produced a slight bump in performance, as did features from @swagician.
By now, I had several CV scores in the 1.028-1.03 range, and public LB scores in the 1.027-1.03 range. A few notebooks had surprisingly low CV scores (< 1.025, with one going to nearly 1.02), but these had LB scores > CV, and had me concerned about overfitting. I also blended with public notebooks a few times, but intended to have at most one such submission among my final 2 (ideally none). Another little bump in my score came courtesy of @noodl35 generously sharing new features generated using OpenFE. Using these ~140 features with Autogluon (AG) immediately resulted in the CV score of AG improving from ~1.045 to ~1.040. Naturally, I added that to my ensemble, and also used these features with several other models.
With the last week approaching, I had many ideas for experimentation, but was running low on motivation, and also had a few responsibilities to take care of. An obvious step I didn't get around to was feature selection from the ~170 features. I also couldn't get around to trying out several models I had in mind, or to try a classifier in hopes of the resulting probabilities helping in identifying/predicting outliers, etc. I did, however, play around with various scoring functions and bootstrapping methods in CatBoost, which added to the diversity of the ensembles. I also tried a few different flavors of neural networks, at least some of which helped as well. 
In the end, my best solo model was a CatBoost with all the features & a 30-fold run (CV: 1.03152, public LB: 1.02935, private LB: 1.03081). In terms of ensemblers, the order of best results had been AG > Ridge > Lasso all along, but at the back of my mind, I knew that Lasso had often produced better scores on the private LB even when Ridge was better on the public LB, so I was wondering which one to choose among my final 2, in case I ended up with a better submission with them than with AG. The reason to anticipate that was the time each method took - Ridge and Lasso took a few minutes, while AG took about an hour, and Hill Climbing was taking several hours once I got beyond a few dozen OOFs. Also my poor MacBook was moaning and groaning under the strain on the last day, and I did end up being an iteration behind with AG. As luck would have it, the final iteration with 81 OOFs saw Lasso outscore Ridge even on the public LB, so my top scoring submission was with Lasso (CV: 1.02729, public LB: 1.02601, private LB: 1.02765), and kept me in the 9th place, where I'd been for a few days. The AG run with the same 81 OOFs finished slightly later (too late to be considered), and scored better (1.02536, , 1.02398, 1.02582). It would have been satisfying to finish with that score, though the rank would have remained the same. Still, I do have the satisfaction of choosing well this time.
Many congratulations to @cdeotte for his well-deserved win with an amazing solution, and to everyone who finished well! And thanks to everyone who shared their code and insights, including but not limited to 

@cdeotte,  @martynovandrey, @backpaker, @swagician, @noodl35, @oscarm524, @ravaghi, @siukeitin 

As I look back upon my year on Kaggle, it's been an amazing ride. I started at the very end of February 2024, and then participated more or less fully in the next 10 episodes, with 8 good finishes:

|Month| Rank/Number of participants|
|-------|----------------------------------------------|
|March 2024| 6/2199|
|April 2024| 9/2606|
|May 2024| 7/2788|
|July 2024|4/2234|
|August 2024| 1/2422|
|September 2024| 3/3066|
|November 2024| 13/2685|
|December 2024| 9/2390|

</br> 

The two remaining months (May and October) were ones where I held the no. 1 for a big chunk of the month (based on the performance on the public Leaderboard, which uses 20% of the test data), but had overfit to such an extent that I fell out of the topmost ranks when the rest of the test data was used for evaluation, though I was still in the top 4% (111/2684) and 8% (299/3858), respectively. 

Along the way, I also got to pick up some AutoML via the Kaggle AutoML Grand Prix, a competition held on the first of each month from May to September 2024. It's been eye-opening to see the advances in this fast-growing field. All this has also led to some Kaggle swag, and whetted my appetite for bigger challenges.

As we start a new year, my goal is to continue participating in TPS, but devote less time, and spend a little more on learning new things, and participating in more prize competitions. I wish you all an amazing 2025, full of new learning and adventures, and may most of your wishes come true (let's leave a few for the years to come 😀). Happy New Year, and Happy Kaggling!
