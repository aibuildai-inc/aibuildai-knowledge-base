# Competition thoughts/source code (#19)

Competition: predicting-red-hat-business-value
Rank: #19
Source: https://www.kaggle.com/c/predicting-red-hat-business-value/discussion/23773

I only worked on this for two weeks - starting after the Talking Data leak/messup.  I wound up building on what I learned about Pandas from DuneDweller’s script, and was able to get to the top 10 with a week to go… and then I sputtered, possibly because of a (viral?) thing that sapped my thinking last week.  Still my first silver medal in a while - and tied with Springleaf for my best score.

The leak was quite interesting and reshaped the data set a lot.  But even if a time split had been used, there still would have been a lot of quasi-leaks as there would still have been many all 0/1 groups in the test set.  It’s not like TalkingData where one looks at the leak and thinks “what so-called randomizer did they use?” - it evolved out of the dataset itself.

The main ideas were to cross verify the leak, and then after I had some reproducibility issues, creating a stratified split on people_id.  With those I was able to get to .9927xx… and not much further.

Interestingly, one group (17304) contains 30% of the dataset and is totally 0 outcome.  I think it’s a group of random non-customers on the Internet or something.

group_1 itself was a perfectly valid feature, from my CV and XGB analysis.  There was some overfitting on it, but nothing major.

edit 21 Sep:  I forgot to mention that a key feature (worth ~.0005) was the # of people per group.  xgb heavily and accurately keyed in on single person groups.

Also there are tons of duplicates, they tended to be 0 outcome but I was unable to implement it in a model in time.  (I bet that was used by some of the top 10.)

My last idea (which I started with less than a day to go!) was to build a stackable model that determined whether a group was likely to be all 0’s, 1’s, or mixed.  I got it down to a multiclass logloss of .735, and it was a very powerful feature - in fact I fuzzed and binned it in the final XGB model that used it!  I was able to get from .992914 to .993118 using it.  Trying to do that with only three submissions left was very nerve wracking, but my last two got progressively better at least.

And my very final thing before the competition deadline was to go through and copy and very slightly (but not enough!) edit the actual jupyter notebooks I used.  I ran them all under kaggle’s Python Docker container on three different machines (i7 2600, dual w5590, and dual e5605)

No, one really *didn’t* need all that aging-yet-still-very-useful hardware for this competition, but I found that xgboost works better with sparse models… but is *very* variable.  So the copy that ran on the 5605 produced a better model, and I submitted that.

And I got bit by the xgb-liking-sparse-matrices thing badly - an entire line of models I was working on was slightly worse on the LB, even after feature engineering.  Between that and my viral? thing, that got me stuck for a bit...

Potentially embarassing code quality aside (I find Jupyter notebooks the equivalent of working on a breadboard), you can look at the notebooks at [my github][1] now.  There’s some nifty use of Pandas, I think.

Now… when’s the next xgb/pandas friendly competition*? ;)  Guess it's time to learn streaming methods...

---

PS... Congrats to all the winners!  I'm looking forward to hearing how raddar got to .995!

PS2 - Getting Bosch's data into Pandas is much easier than I expected ;)

  [1]: https://github.com/happycube/chadslab/tree/master/kaggle/redhat
