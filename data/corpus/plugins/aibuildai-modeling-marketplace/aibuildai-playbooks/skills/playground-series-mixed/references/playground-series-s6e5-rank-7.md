# 7th place solution

Competition: playground-series-s6e5
Rank: #7
Source: https://www.kaggle.com/c/playground-series-s6e5/writeups/7th-place-solution

This is my first month of serious attempt at a playground series, so I put in at least 1~2 hours of work every day. There are still many things which I am unsure whether I did correctly.

# Base models

~25 models I built myself and from various public notebooks.

50 autogluon models from "best_v150" preset, using a baseline set of 18 features - 4 categoricals (I make Year categorical), 10 numericals, plus "max_laps" ("Lap_Number" / "Race_Progress", a very strong feature), "TyreOvrLap", and count encoding of "Driver" and "Race".

101 autogluon models from "best" preset, using a baseline set of 18 features.

This gives 176 models in total. I found autogluon to be a cheap way to add a large number of diverse models.

# Ensemble

Then, I train a stacking model on autogluon using those 176 OOFs as features. This gives a public CV of 0.95453. An equal weight ensemble of this and the popular CV=0.95454 blend gives CV=0.95463! So my own model is quite diverse from the public blend.

In the last few days, I did some more experiments on emsembling. The final submission is as follows:
```
Submission 1
Weight 3: Public CV=0.95454 blend
Weight 1: Stacked ensemble of 7 stackes + final hillclimbing on those 7 stackers. Use a Logistic Regression fitted on 176 OOFs to select 10% of test samples for pseudo labeling and add them as training samples to the stackers.
Weight 1: Stacked ensemble of 50 stackes + final hillclimbing on those 50 stackers. No pseudo labeling.
Weight 1: Stacked ensemble of 50 stackes + final logistic regression on those 50 stackers. No pseudo labeling.
```
```
Submission 2
Same, but change weights to (2,1,1,1).
```
In the second submission, I lower the weights of public CV because I don't trust those "blend of blend" that much. This give 0.1 bps lower public LB score, but 0.1 bps higher private LB score compared to submission 1.

# Special Thanks

Finally, I would like to thank everyone who posted in the code and discussion section. The following are particularly useful:

https://www.kaggle.com/code/yekenot/ps-s6-e5-realmlp-pytabkit RealMLP is the strongest single model in this competition.

https://www.kaggle.com/code/raunakdey07/f1-pit-stops-blender-0-95454 blend I used

And special thanks to @cdeotte for all his previous solutions in the playground series! Those solution write-ups are the single biggest help for me in tabular machine learning. If you havent read them yet, make sure you do! In particular, this one has been very helpful:

https://www.kaggle.com/competitions/playground-series-s6e3/writeups/1st-place-gpt5-4-gemini3-1-claudeopus4-6-kgm

In my opinion, S6E3 is very similar to S6E5, and deep stacks work well in them. S6E4 however is different, since the target is noisy (has a sparse class).
