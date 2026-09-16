# 8th place solution (team LAJMBURO)

Competition: bosch-production-line-performance
Rank: #8
Source: https://www.kaggle.com/c/bosch-production-line-performance/discussion/25382

Below is the solution of team LAJMBURO (8th private LB (0.50888), 11th public (0.51001) ).

There are two audiences intended:

* If you are interested in only the models/features, then read the part 1 until "Features" included and you can skip everything else (maybe you might want to read "Validation method" and "Submission method" from part 2 of this post.
* If you are interested at how a team of 8 is working, then read from “the name of the team…” which is part 2

---

## Part 1: Models & Features

Two parts:

* Modeling and data
* Features

### Modeling and data

We used simple **level 1 models**:

* LightGBM
* xgboost
* Random Forest
* Neural Networks (didn’t get picked up on level 2, so we removed it)

We had **five level 1 datasets** (gbm=LightGBM, xgb=xgboost, rf=Random Forest - features are not exactly including all the features described later), I’m including their **mean MCC on cross-validation** (for Public LB score, add at least 0.02 - numbers might be off a bit):

* Data set 1 (0.477 gbm): order, raw numeric, date, categorical
* Data set 2 (0.482 gbm, 0.477 xgb, 0.473 rf): order, path, raw numeric, date
* Data set 3 (0.479 gbm, 0.473 xgb): order, path, numeric, date, refined categorical
* Data set 4 (0.469 xgb, 0.442 rf): has features sorted by numeric values + date features + faron’s magic features, path, unsupervised nearest neighbors (L1 = Manhattan / L2 = Euclidean distances) per label
* Data set 5 (0.43 xgb): has faron’s magic features, path, unsupervised nearest neighbors

**Level 2 data set** was a bit special, and included:

* Level 1 predictions (we had 12 predictions from level 1)
* Data set 5
* Duplicate feature (count and position)

**Level 2 stack models** (giving the weaker model a stronger weight was better):

* 30% weighted xgboost gbtree (~0.488 CV)
* 70% weighted Random Forest (~0.485 CV)

We had a very **minor issue at overfitting LB**:

* We used a threshold multiplicand of 0.92 from the cross-validated threshold (which seemed the best multiplicand), 0.51001 Public LB, 0.50888 Private LB
* If we were using the threshold from cross-validation without a multiplicand, we would have had, 0.50923 Public LB, 0.50954 Private LB. But that's not an issue at all overall.

In the case it were to overfit severely, we also made a backup set which is leakage-free (using features not using the label directly) and without a multiplicand. For instance, features like the Unsupervised Nearest Neighbors was removed. A single xgboost provided 0.506 on Public/Private LB, good enough for a gold medal.

### Features:

**Order**:

* Faron’s magic features (sorted by max time, min time)
* Magic features sorted by numeric values
* Same timestamps previous/next sample after sorting by tmax
* Previous / next response for duplicates

**Path**:

* Redefine station and line numbers based on timestamps to make sure timestamp was constant for each station
* Cluster of samples by path and calculated per sample the absolute and relative difference between numeric / time difference (max - min) features and cluster mean. 
* Path based error rate (leave one out method) but it did not improve score
* Merge features based on path
* Entry / exit station
* Previous and next station for S29 - S37 (one-hot encoded)

**Datetime**:

* Timestamp per station, per line, merged per path
* Max-min per station, per line, merged per path
* Kurtosis + kurtosis per line (very strong, surprisingly)
* Lead/lag response rate statistics after sorting by tmax, id
* Lead/lag response rate statistics after sorting by tmax, numeric, id
* Timestamp based error rate (leave one out method) but it did not improve score
* Timestamp label-based density (overfit)

**Numeric**:

* Use supervised decision trees to try to predict the numeric value using the date, per station => the split points allow to define thresholds, which clustered the values by group of dates (over 4000 features) => used xgboost per station to shrink it to ~120 features as adding 4000 features didn’t help at all
* Numeric after removing time trend (did not improve score)
* Raw numeric data (a specific selection of them)

**Categorical**:

* Couple of features one-hot encoded added a little bit

### In the end

Earning 0.500+ LB is possible in 10 minutes using a single model (LightGBM) under 16GB RAM with appropriate features. But creating these features do need more than 16GB RAM…

xgboost can outperform LightGBM if tuned appropriately (my tuned xgboost was scoring 0.483 CV, which confirmed itself on LB, beating all our cross-validated model submissions including LightGBM) but it is much slower than LightGBM! (and was eating 58GB RAM.. mmm yummy! when you know LightGBM used only 16GB at most!)

---

---

If you are not interested into reading about the team (and only were interested in the model/feature), you can skip everything below (there might be the "validation method" and "submission method" which you might want to read).

---

---

## Part 2: The team

Multiple parts:

* The name of the team
* Team composition (this is what happens when you are play the recruiter for a team project)
* Validation method
* Stuff we used in the team

### The name of the team…

It is just the concatenation of the initials of the team members! We didn’t have a hard time to find it… because I put it temporarily as a placeholder name. And then we didn’t even chat about the name...

### Team composition

I had so many people applying after my Red Hat forum post that I had to make a selection. For a team of 6 (that's big for Kaggle - but "big" is not that big, really) working like in a real project, it requires a lot of balancing and diversity in skills. Therefore, using the 2 to 20+ lines people sent me when “applying” (either directly by email for those who got my email, through Skype, through text, or through Kaggle), I selected 4 people (I had already one in my team) according to their skills:

**Finally, we had:** (including a diversity of R/Python)

* One who can go “ham” for trying to push on LB (Shubin) - great competition spirit
* One with a lot of “let’s try this” features (Joost) - it helped a lot
* One who can analyze methodically what he has and what he gets as feedback to improve (Michael) - very good at avoiding going too fast
* One who can help providing better insights using R packages over Python packages (Rodolphe) - great at providing alternatives
* One who is more about the organization and providing a good kickstart with sample test scripts for the team (Alex, cf see [what][1] he can do in the House Prices competition)

And from my experience in making hackathons with mandatory teams, I do know that learners:

* Are starting very fresh in mind and are extremely committed early
* Are (not always for any person) slowly dropping out as time passes by and commitment requires to increase (which is exactly what must not happen when kaggling)

In addition, we all have a RL (a real life) out of the world of Kaggle, therefore availabilities could be somewhat scarce and rare.

Hence, typically we were 2 to 4 working on this competition. Thus, that’s not really a big team, and it was extremely easy to handle.

There are two team events that happened:

* Adding jayjay to our team, when he asked for a team (while he was in top 15 Public LB with 0.49+), as he had a solid feature set (we knew it could be a bit overfitting before even merging with him, as our team experienced massive differences by just changing the threshold for MCC)
* Adding onthehilluu to our team, to think more about pushing on LB (feature insight) and sharing knowledge/experience (when we merged with jayjay and managed to push a bit on LB with his single model, the initial objective of sharing experience was left behind, and the new objective was set to get a gold medal, i.e top 12)

And frankly, we were stuck with jayjay’s set for about 15 days. Literally, it took us that time before we started to see real improvements: there is a strange feature conditioning which made all our sets we had to not play well together (by pair - with jayjay’s). When we started to add new features (not from any existing feature set we had), real improvements kicked in. And there was only 1 week left when we started seeing good improvements…

### Validation method

We used 5-fold cross validation for our supervised models. I generated a fold very early during this competition so we could all have the same basis to work on this data. But then we found my folds might not be stable enough! Therefore, someone created “stable folds” to use. However, what we found is that “too stable” is “not good enough”. The issue was probably lying in the duplicates. Therefore, we decided to stick with my initial folds, which are actually very good as we could spot fold instability very easily!

### Submission method

This one is a very long text to explain. There is no magic behind submitting (compute threshold from CV), but not wasting submissions is another thing. How to submit on the LB was not very tricky. Not tricky due to the metric to optimize: either 0 or 1s. Add into the factors the fold instability... That was.. literally doom. I worked on a “submission creator” which took CV prediction (out of fold), CV test predictions, and a full model predictions, and which output a lot of different metrics in addition to a diagnostics file and 10 submission files.

Each of these 10 submission files had their positive count printed in the diagnostics. Here is an example of diagnostics.

     Fold 1 converged after 0836 iterations.
     Fold 2 converged after 0571 iterations.
     Fold 3 converged after 0499 iterations.
     Fold 4 converged after 0585 iterations.
     Fold 5 converged after 0566 iterations.
     Iterations: 611.40 + 129.874
     
     
     Fold 1: AUC=0.9125954
     Fold 2: AUC=0.9221191
     Fold 3: AUC=0.9188253
     Fold 4: AUC=0.9200112
     Fold 5: AUC=0.9183175
     AUC: 0.9183737 + 0.0035463
     Average AUC using all data: 0.9037691
     
     
     Fold 1: MCC=0.4756098 (0534 [38.84%] positives), threshold=0.3749080 => True positives = 76.592%
     Fold 2: MCC=0.4769881 (0437 [31.76%] positives), threshold=0.5057880 => True positives = 84.897%
     Fold 3: MCC=0.4801341 (0560 [40.70%] positives), threshold=0.3444360 => True positives = 75.536%
     Fold 4: MCC=0.4737958 (0605 [43.97%] positives), threshold=0.3298900 => True positives = 71.736%
     Fold 5: MCC=0.4829743 (0615 [44.69%] positives), threshold=0.3086210 => True positives = 72.520%
     MCC: 0.4779004 + 0.0036627
     Threshold: 0.3727286 + 0.0781904
     Positives: 550.20 + 071.37
     Detection Rate %: 39.991 + 05.185
     True positives %: 76.256 + 05.237
     Undetected positives: 0959.20 + 0028.86
     Average MCC on all data (5 fold): 0.4734361, threshold=0.3727286
     Average MCC using all data: 0.4747755, threshold=0.3478090
     
     
     Submission overfitted threshold on all MCC positives: 2896
     
     Submission average validated threshold on all MCC positives: 2736
     
     Submission average of overfit+validated threshold positives: 2814
     
     Submission with all data overfitted threshold on all MCC positives: 2933. Threshold=0.347809
     
     Submission with all data average validated threshold on all MCC positives: 2803. Threshold=0.3727286
     
     Submission with all data average of overfit+validated threshold positives: 2867. Threshold=0.3602688
     
     Submission with all data by taking the amount of positives of overfitted threshold on all MCC positives: 2896. Threshold=0.354801
     
     Submission with all data by taking the amount of positives of average validated threshold on all MCC positives: 2736. Threshold=0.385727
     
     Submission with all data by taking the amount of positives of average of overfit+validated threshold positives: 2814. Threshold=0.370321
     
     Submission with all data by taking the sum of positives of validated positives: 2751. Threshold=0.383094
     
     Cross-validated used features list (all used features to copy & paste):
     
     c("sameL0_next", "sameL1_next", "CATEGORICAL_Last_____1", "S24.311", 
    "sameL3_next", "GF1", "GF0", "FOR30_Sum_S", "CATEGORICAL_out_out_L3_S32_F3854_class2", 
     (censored because way too long)
     
     Cross-validated multipresence used features list (all used features to copy & paste):
     
     c("sameL0_next", "sameL1_next", "CATEGORICAL_Last_____1", "S24.311", 
    "sameL3_next", "GF1", "GF0", "FOR30_Sum_S", "CATEGORICAL_out_out_L3_S32_F3854_class2", 
     (censored because way too long)

Does it look like Chinese to interpret what is wrong and what is OK? At first sight, yes. I made a small tutorial (by message on a chat… un-obviously) to help my team (those who are modeling and/or testing internally features) decipher the diagnostics file. Here are good excerpts of what I wrote:

> My folds have extreme feature variance on each fold (not intentional,
> but it seems to help a lot), so if your features don't work/generalize
> on several folds, it will decrease the stability.
> 
> CV variance is OK, but too much variance is not. I tested over 20
> combos on 3-4-5-6 CV folds locally to test how variance can increase:
> 
> * at 6 fold, standard deviation (sd) is about 0.01
> * 5 fold, sd about 0.006
> * 4 fold, sd about 0.006 (not sure?)
> * 3 fold, sd about 0.004
> 
> At the top of the diagnostics, you have all diagnostic things to
> analyze the model, so you can check for consistency "out-of-fold" (out
> of fold predictions) and "all-of-out-of-fold" (when taking all out of
> fold predictions together): AUC, MCC, threshold, positive rate,
> detection rate, true positives, undetected positives (false
> negatives), MCC using 5-fold average threshold, MCC using all data.
> 
> If the difference between both is high, most of thresholding methods
> we use will fail (and we have a fallback for this in this scenario).
> For instance, 0.92 avg AUC out of fold, 0.85 AUC all data is not
> stable for submission. You can have high AUC and "trash" MCC. But we
> use AUC to monitor if predictions remain stable on each fold so we
> don't get any surprize when training on all data (because from that
> point we go nearly "blind"). If your probabilities predicted are
> stable, AUC on all data should be close to the average per fold
> 
> When your folds are unstable (due to XYZ feature), the difference
> increases as a [0, 1] scale for a fold might not be much more
> different against the same [0, 1] scale for another fold.
> 
> My script creates 10 submissions:
> 
> * 1st one: use threshold using all data, on CV averaged prediction = mostly for debugging purposes
> * 2nd one: use average threshold from CV, on CV averaged prediction = mostly for debugging purposes
> * 3rd one: take mean of 1st and 2nd, on CV averaged prediction = mostly for debugging purposes
> 
> Then you have 3 more submissions created (4th, 5th, 6th) which uses
> the predicted probabilities on testing data using the same
> thresholding techniques of 1st, 2nd, and 3rd. By default, the 4th
> submission is the best to submit (it's not that "overfitted" - it
> overfits more and more if your fold stability is lower)
> 
> Then you have 7th 8th 9th, which uses the positive count found using
> the 1st 2nd 3rd methods but on the predictions using the model with
> all data together (those one should be avoided most of the times, but
> they are left as is if needed).
> 
> And the final one the 10th: it takes the positive count of the CV and
> uses that same count on the test data from the model using all data.
> It assumes the hypothesis the positive count in train = positive count
> in test, which might not be true. Obviously, no one knows if true or
> not. This is the submission you must use when your folds are clearly
> unstable (it is the fallback method for submitting with an unstable
> CV).
> 
> If you see on these debug a number like 10,000 or more (usually it
> sits around 2000-4000) it means your folds are unstable (captain
> obvious is here) and you need to look out what happened. It means also
> your CV models / full model predictions are unstable too (so only 10th
> should be used, unless the one you pick has stable predictions = value
> around 2000-4000). Therefore, only the fallback submission should be
> valid from that point, in most cases.
> 
> If you make a submission using 1st, 2nd, or 3rd set, you MUST
> approximately get what you had in CV. If you don’t, the feature you
> added is bad enough to be conditioned by something in the folds. When
> you want to test 1 or more submissions from a single model, you use
> this order if your model is stable according to the diagnostics and
> your sight of it:
> 
> * 4th (nearly always the best)
> * 6th (rare you need 2 submissions - only if you clearly think your model is good - acts as a confirmatory response vs your CV if 4th
> overshot the LB)
> * 5th (very rare you will need 3 submissions to test a single model - typically happens when we don't have enough to submit, this should
> always score lower on LB than 4th/6th no matter what you do, unless
> luck)
> 
> If unstable: - 10th (and only this one)
> 
> Average MCC on all data (5 fold) <- use 5 fold threshold, average, and
> compute MCC using that average threshold
> 
> Average MCC using all data <- compute threshold on all data, use that
> threshold on all data
> 
> Cross-validated multipresence features are in case you have features which are not picked up by your model if you input LightGBM CV output from my package. You can't compare both using `featurelist1[which(!(featurelist1 %in% featurelist2))]` in R.

### Stuff we used in the team

Mainly the most important names for a fully decentralized team working on a Kaggle competition in a (non or not-non) project-mode:

* R programming language
* Python programming language
* [LightGBM][2] through the [Laurae package][3] (fastest R implementation at that time - you can do a 5-fold CV on 1000+ features on less than 40 minutes, I/O included)
* [xgboost][4] (because “unidimensional machine learning at Kaggle is not Kaggle without xgboost”)
* scikit-learn Random Forest (eats RAM but it works in Python without having to go through Java)
* H2O Random Forest (when in R you want to use parallelized and fast Random Forests…. T_T)
* Keras Neural Networks (who needs a neural network in the brain?)
* data.table (fread/fwrite and super fast data manipulation, please)
* Markdown, Rmarkdown, hacked pandoc (welcome Visual Studio executable hacks, by Laurae), iPython notebooks (yea, yea, we did reporting in the team to share insights / do “bookkeeping” - in a real project-mode)
* Computers (who does not need a computer to compete at Kaggle? Use a phone?)
* Monitors (because we are not blind)
* Windows / Linux (because you need an operating system on your computer…)
AWS / Google Cloud Platform (when you needed more horsepower on your car)
* SSH FTP & Filezilla (how do you share data securely with the 1Gbps network from Laurae with all the team, when you have 10GB files?)
* RStudio / Spyder (who needs an IDE to program?)
* Remote Desktop (to share sessions on a server, troubleshooting, etc.)
* Skype / Phone (could use Slack too, but we didn’t in the end as Skype was faster)
* GitHub (private repository, to share code privately)
* An internet browser to go on websites (it’s obvious you need to use a browser somewhere)

The team experience so far? Really great! So much fun and no tension in the team, with people listening to each other => a nice ambiance to spend time on Kaggle. And people wanting to learn from others is always a wonderful experience, as it is the **TEST/challenge which checks whether your knowledge was correct or not correct / biased** (yea, you can learn wrong something, it may happen).

--- 

Ah, and I think this is the post which will put me to Master tier in Discussions. 199 medals, now 200. A good double Master tier from this competition. Anyways, great work from everyone in this competition! (not only my team but the others! - look at all these great kernels and especially the pathing one!)

There are 5 new competition Masters according to the top 12 private LB! (Alexey Noskov, Laurae, Shubin, Michael Maguire, Alexandru Papiu) - congratulations!


 [1]: https://www.kaggle.com/apapiu/house-prices-advanced-regression-techniques/regularized-linear-models
 [2]: https://github.com/Microsoft/LightGBM
 [3]: https://github.com/Laurae2/Laurae
 [4]: https://github.com/dmlc/xgboost
