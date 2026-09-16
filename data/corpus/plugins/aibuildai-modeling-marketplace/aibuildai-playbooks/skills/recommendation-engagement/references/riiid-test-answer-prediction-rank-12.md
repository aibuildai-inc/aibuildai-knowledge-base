# 12th place Solution

Competition: riiid-test-answer-prediction
Rank: #12
Source: https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/209635

[riiid-stack]

First of all, I want to thank both the hosts(kaggle and riiid) for hosting this competition.
It was a really challenging competition in many ways, and really tested my limits. 
The competition was also very clean (albeit the privateLB bug unrelated to this competition). A very clean and logical train/test split/correlation is what makes this competition shine.

---------------------------------------
# Overall solution:
For our single mode, NN scored around 0.806-810(privateLB), LGBM 0.804.
We stacked these single models with NN and LGBM to achieve a score of 0.816.
I suspect our single models were mediocre in terms of score, but the stacking got us to gold.

Regarding our 5place drop on the privateLB, we kind of expected it. We knew that one of our models(SAINT) had a bad private score because of the private score leak, but we couldn't quite fix it. Even after the competition has ended, we still have no clue as to why the model suffers in the privateLB.

---------------------------------------
# Validation:
We used tito-split throughout most of the competition period.
We mostly experienced good Valid-LB correlation, which I believe is the same for others. The single models used a 95M/5M split, and the stack used the 5M for training/validation.
We also tried user-split as our final submission for diversity, but this had no impact on our final score.

---------------------------------------
# LGBM details:
#### Resource management
(When generating features for submission)
I used h5py for memory-hungry features like (user x content features, user x tag features).
The h5py file uses user_id as the key, and read the whole user-feature when hitting new users while predicting.
This creates a good balance between memory and runtime. Since there are not many unique users in the test-set (compared to train), the time to FileIO is not too much, and huge amounts of memory is saved.
All the other features were pickled and loaded to memory in the beginning.

#### Features: 
Question features:
These were made by the full training-set(100M). 
Obviously this is leaky, but since every single question has >1000 counts, the leak is tolerable.
Example features: mean question ac, mean user rating of who did not answer correctly.

User features:
How good the user is, especially related to parts, tags, contents.
Example features: user x contents mean question ac

Timestamp features: 
This was kind of a surprise to me. Not only was the timestamp diff of t and t-1 a good features, diff of t-1 and t-2 up to t-9 and t-10 improved my model.
Example features: user user timestamp diff from last lecture

Rating features:
Elo features from this [notebook](https://www.kaggle.com/stevemju/riiid-simple-elo-rating) (thank you very much). Trueskill did not improve my model.
With only mean ac, the model cannot determine if the user is challenging hard questions or easy ones, so rating questions and users makes a lot of sense.

SVD features:
LGBM is bad at expressing category columns (compared to NN). So I took the question embedding layer of NN, and used the 20dimension SVD as features.

#### Feature selection:
I used about 70-80 features for my model.
Since we were stacking a lot of models, I didn't want to use too much resource with my LGBM(runtime, memory), so I picked features which had a lot of impact, and made my model contribute to the stacking-model. There were a lot of features which didn't improve my model much, and all of them were thrown away. 

#### Hyper-param:
I didn't change this too much, but increasing the num-leaf 127->1023 improved my score by 0.003, which was a surprise. This happened after I added lots of timestamp features, so there might be a very complicated interaction underlying in the timestamp.

#### Machine:
I used GCP, 64coreCPU 416GBmemory. Even with this monster machine, I ran out of memory a lot when generating the full features (which I avoided by processing in chunks).
I believe this instance cost me around $1000 over the competition (this is all payed by my company, and we are hiring btw). A lot of this cost happened because I was lazy (never used a preemptive instance), and I believe you could still be competitive with a $100 budget, even on this HUGE data, so don't be discouraged by the cost if you are a Kaggle beginner (I would recommend a competition with smaller data though :) )

---------------------------------------
# Stacking details:
Stacking is always difficult. It often leads to overfitting.
In this competition, there was a very good Valid-LB correlation, so I guessed(correctly) that stacking could work.
I wanted to make sure this layer works, so I tried to do everything conservative.

#### Validation:
3M/2M train/valid setup. For the final subs, I made multiple models with a time-series-split (negligible gain).

#### Input Models:
Multiple models from Sakami(SAINT based and AKT based), which had different window size. We couldn't fit in all the models, so we chose 3 as our final sub.
Owruby had another SAINT model which both added diversity and improved the stack a lot. Lyaka had a SSAKT model which also helped slightly.
I also added my LGBM model, which surprisingly improved the stack, even with the features added.

#### Input Features: 
LGBM stack model: Hand selected 15 features from my single LGBM model. There was literally no gain from the other 65.
NN stack model: Selected features + the last layers from singleNNs. This was done by Lyaka.

#### Stacking models:
LGBM stack model: num_leaves was reduced 1023->127. 
NN stack model: Lyaka did both MLP and a Transformer. Both had similar scores, and we used MLP as the final sub.

#### Final output:
Mean of LGBM and NN stack model

---------------------------------------
# Final thoughts:
Although we couldn't quite achieve the goal we wanted (beating mamas), I think we tried our best and did our best.
Everyone contributed to the final stack, everyone did a ton of work, and we had great teamwork to make the stacking happen. I am really happy with the team we had, like I always had throughout my kaggle history. Thank you all.
