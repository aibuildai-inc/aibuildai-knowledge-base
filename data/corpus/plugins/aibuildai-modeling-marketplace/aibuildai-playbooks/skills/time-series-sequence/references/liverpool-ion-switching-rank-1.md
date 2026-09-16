# 1st (or 16th) place short write up

Competition: liverpool-ion-switching
Rank: #1
Source: https://www.kaggle.com/c/liverpool-ion-switching/discussion/153940

Hi Everyone,
Thanks particularly to Chris Deotte, Vitalii Mokin, Trigram.&nbsp; I'm sorry to forget people, we'll spend the next days upvoting things we've neglected.&nbsp; Thanks to the organizers and the kaggle team, this whole competition meant so much to us, and just getting to gold medal level was a great feeling.  

Concerning the leak, it's been well described already but there were other clues, and we certainly didn't think we had something new. &nbsp;
- The Matlab code has a vague comment about not working above 5
- It isn't seeded with the clock. &nbsp;It's a "feature" of Matlab (not a bug) that the RNG by default produces the same chain on startup. &nbsp;This was probably the original source of the problem.
- There are visible beats in the 50Hz of the max10 set. 
- You can actually see the correlation in the signals visually (we happened to be adding 5+5 to get some extra noise for simulation and checked .corr across everything. &nbsp;It looks like lots of people had the same idea here)

The leak was entirely limited (as far as we know) to one max10 private set. &nbsp;Of course everything seems obvious once you know it, but we expected a small advantage if any and a big leaderboard shake that everyone was talking about. &nbsp;I didn't think it would be at all appropriate to "disclose" it in the last week.

Take this as the first place rundown, or the 16th place rundown as you prefer.  We will eventually provide more details, but I didn't plan on having to write anything up so soon. 

One large improvement came from 50Hz cleaning. &nbsp;We ended up averaging signal minus lgbm predictions over a few cycles after labeling them mod200. &nbsp;This nicely accounts for small drifts around 50Hz in both frequency and amplitude, and is incredibly simple. &nbsp;(a real fourier transform wasn't the right way to go for us).

The shifts were also obviously important, as Rob pointed out recently. &nbsp;His fourth point also can't be emphasized enough. &nbsp;

We spent a *lot* of time on a bayesian inference model, although we didn't use it in the very end since the max10 set suddenly became max5, and bayes was virtually identical to LGBM after good 50Hz cleaning (but free from any machine learning!). &nbsp;This required sufficient monte carlo to find state transition probabilities that we also used for data augmentation.

Finally, the max25 set was initialized differently (the second "low" HMM). &nbsp;This turns out to be important for a Bayesian model, as we began with the initial state and updated backward with each signal point. &nbsp;(although it's not a significant impact for f1)
