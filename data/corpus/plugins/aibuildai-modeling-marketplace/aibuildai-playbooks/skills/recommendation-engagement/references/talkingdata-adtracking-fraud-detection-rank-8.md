# 8'th place solution. [ods.ai] blenders in game )

Competition: talkingdata-adtracking-fraud-detection
Rank: #8
Source: https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56325

At first, as usual, I want to say Thank you to all kagglers who share their approaches and thoughts during this and other competitions. Despite some annoying events (unfortunately such events can happen in any competition) kaggle platform is still the best place for practitioner Data Scientist who want to lift up their skills and go deeper into this DS Rabbit's hole )

Our Team was built last week just before the end of competition (before merging deadline), I was at 80th place (at that moment), @johnpateha just decided to take part in this game that day, @ppleskov, @maksimovka, @yaroshevskiy and @ddanevskyi occupied ~100th places on the LB. I was ready to get a bronze (in case of luck) and finish my attempts to climb up, but one morning...

"Knock, Knock, Kruegger" ) say @johnpateha. "We have a good chance to catch a gold here, just trust me, wake up and go **"вджобывать"** (to do a really hard work). "The metric is fine, there is a lot of data, and in the worst case we don't lose anything".

Hmm. Why not? We made the team, several hours later combined our team with @ppleskov &amp; Co, and start **"вджобывать"**.

**Here I just describe my part of solution, my colleagues will add their own approaches later.**

This competition is really hard due to size of the dataset. We need a lot of RAM to train our models especially if we have a lot of features. So this is why my solution before merging has been based on day 9 for training, day 8 for target encoding calculation - and 10% of train (shuffled) for validation. It was enough to climb up to 9805 score, but I feel that for this pipeline it is a ceiling.

**Dataset**:  

In the final solution I use day 7 for target encoding, day 8+9 for training and last 2.5M rows from train as holdout. We decided to just blend our final solutions and not to use stacking on that level. (and of course, we had to comply with our team name! :)

I spent a lot of hours and made a lot of attempts to fit my dataset to memory, I used all tricks I knew before and found there on the forum - but my machine (with 64Gb RAM) time after time told me "Out of Memory", "Out of memory"...

The final approach that I found - is to use numpy memmap as data storage, I describe this method here:

https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56105

**Features**:

I have several group of features, most of them you can find in the public kernels, nothing special:  

* Count by several groups
* NextClicks
* TargetEncoding over groups
* Statistics (mean/var)
* and so on...

After merging I added to the dataset some features from my colleagues' solution, like duplicate orders, and so on. No "Killer features", but overall score became more stable.

**Feature Engineering**

At first I tried to use my favorite method of feature selection (random shuffling - like Boruta) but without success, so I returned back to old-fashioned style - greedy forward selection by small groups (3-5 attrs at a time). If my val score was raising I added that group to the dataset, in other case - gave up the whole group.

After adding a group I tried to "cut tail" of features based on feature important but in very "conservative" manner.

The final solution contains 74 features.

Finally I selected three big groups of features in addition to the full set and built models for all my best parameters on these sets.


**Models and parameters tuning**

I used lgb as my base model, tried to build FM-FTRL/XGB, but with no overall improvement. I also was unlucky in CatBoost - it refused to run on my data, I don't know, probably it took offense ).

I tried to use some parameters I found in the public kernels, but the best one I got - I got from Bayasian optimization process. I really a big fan of this method so I suggest trying it if you haven't tried it before.

I use this implementation and it gives me very nice results in all competitions I use it.

https://github.com/fmfn/BayesianOptimization/blob/master/README.md


**Diversity in dataset**

Initially I built my models on last 75M rows (according to memory constraints), than on last 100M rows (thanks to my teammates for one more power computer) and finally (when I implemented dark numpy magic :) ) - on the whole dataset.

**Ensembling**

I used method that dropped me down to ~1500 place in the Toxic competition (I didn't prepare it well in Toxic), but in this competition it gave a huge improvement. The method is Scipy optimize with softmax restriction over weights. I tried to use power ensembing, caruana's hillclimbing, just geometric averaging - but scipy (in case you have holdout prediction for all of your model) gave the best score in all environment I tested it.

My final solution was blending with scipy weights over 7 best models (from ~30 overall).

The oof score of my best has .99235 -&gt; .9823x on public LB. Combining it with models from @johnpateha and other teammates we had the final score that lifted us up to 8'th place )

**Final words**

Thank you to all my teammates, especially @johnpateha who pushed me to Gold medal ) We did a very nice cooperative work inside our team, got the results in short time, so my dear colleagues - you are the best! )

It was my first team competition (I played solo before) and I really appreciate the results of working in a team.

Happy kaggling!  
(C) Kruegger

P.S. Some Intrigue - @johnpateha did huge investigation over data, one of his advices helped to lift score of my model up to +0.0001. Advice was "just put 0 to this 4(four)! rows in the dataset." Dark magic? ) But let him to describe his solution himself.
