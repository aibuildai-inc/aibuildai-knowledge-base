# Summary of things I've learned in get to my moderate 13th place

Competition: facebook-v-predicting-check-ins
Rank: #13
Source: https://www.kaggle.com/c/facebook-v-predicting-check-ins/discussion/22189

First I want to say I’m humbled by the profound wisdom I’ve seen in the forum and in the scripts. Doubtlessly the level of knowledge and insight sharing in this competition far exceeds what I have experienced in other events and like many people, I have learned a lot through contriving better models by combining my own ideas with what I’ve learned from you guys. My appreciation goes to the competition hosts and fellow participant demonstrating extraordinary expertise in solving practical and valuable challenges from different angles.  In this post, I intend to first summarize what I have learned in developing my own solution, and then discuss a few better strategies I have read after the competition was over.  I hope to discover new ways to improve my own methodology and will certainly be especially happy if this could be of help to other people as well.

**What I have tried:**
Initially, I thought of sorting place_ids into groups, and building one model on each of these groups for prediction, and combining the predictions if samples are predicted by multiple groups.  This seemed more intuitive to me than dividing the whole feature space into grids and building models in individual grids, since the distribution of training samples seem to be pretty homogenous over the whole place.  Alas, my scheme did not work out because of two reasons:

1.	There are too much overlap between feature space covered by place_id groups leading to a large number to evaluation on each individual samples both on training and on testing.  Eventually the running time on my i7 desktop well exceeded what I would tolerate for a first model, especially when combined with the following point 2.
2.	Rooted in the same overlapping issue, the combination of predictions from different groups became a problem.  Unlike combining results from different models which will usually vote to a better solution, the combination in this situation mixes only one potentially correct prediction with noise.  This only leads to corruption of predictions.  I tried several difference strategies including prediction probability voting, population size weighted voting and Naive Bayes voting, none of these yielded satisfactory result on the subset of the groups I made for testing the models.
I gave up on this method early.  

**Lessons learned:**
Quickly test methods that seem promising, but make sure you test it as thoroughly as you can think of.  If it does not work, find reasons why.  If the reasons are intrinsic to the method, give it up quickly unless you will be able to alter it enough to amend the intrinsic problems.

I then reconsidered the strategy of divide and conquer based on grids in the feature space.  There are a lot of people on the forum already adopted this strategy before I started seriously thinking about it, and there were a significant amount of feature engineering and transformation shared on the forum.  I tuned my models by testing the importance of different features and putting time relevant weights on training samples.  And of course the model parameters are also tested and tuned.  In addition to filtering out the rare place_ids for each grids, I also filtered out outlier points for each place_ids by considering the distance to the place center measured by standard deviation.  I got three models, random forest, xgb, and kNN, each of them scoring between 0.58 and 0.59.  By stacking these three models I got my best score of 0.605 on private LB.  The following strategies helped in improving stacking scores:

1.	 Select more than 3 candidates from each model for each sample.  I chose 6.
2.	Use probabilities from each models for combining the prediction instead of uniform voting or weighted voting based on public LB scores (between the latter two, score weighted voting worked better than uniform weighting).  
3.	The probabilities from each model do not carry absolutely the same meaning. A straight forward probability voting yields less than optimal scores.  I scaled the probabilities according to the mean probabilities of the top candidate for each models.  That is, after scaling, the mean prob of the No. 1 candidate from each models are the same.  This yielded the best voting result for me.  Biasing the probability slightly to the better models (according to the public LB scors) on top of this did not help.

**Lessons learned:**

1.	 Feature engineering and transformation are important, as is known to all data scientists.
2.	 For each model, take care to obtain the best performance.  I did not have enough time for further tuning the xgb model, which was quite time consuming as I did not really have much free time.
3.	 Develop as many reasonable models as is permitted, starting from the simplest.  I would like to include Naive Bayes and neural network and SVM and KDE if I had time.
4.	 Stacking models together is powerful.  Try to combine models as diverse as possible, and also consider varying within the same model and combine these variations before further combining with other models.

**Discussion**

It is especially beneficially to read leading solutions, try to learn insights and discover my own weakness.  Particularly I want to discuss the solutions provided by Tom Van de Wiele (https://www.kaggle.com/c/facebook-v-predicting-check-ins/forums/t/22081/1st-place-winning-solution), Jack (https://www.kaggle.com/rsakata/facebook-v-predicting-check-ins/3rd-place-solution-simple-version/code) and Markus (https://www.kaggle.com/c/facebook-v-predicting-check-ins/forums/t/22078/solution-sharing, in one of the replies) .

These solutions exhibited quite divergent styles in problem solving but they share at least one commonality:  understand the problem, model in depth, and take care of details

Tom’s model took several stages.  In two steps he narrowed down to 20 candidate place_ids for each sample, and in another two steps and he used extensive feature engineering to pick the top candidates by modeling binary predictions for each place_ids.  When I began the competition, I “intuitively” thought this kind of extensive modeling would be of insane time complexity.  I was very surprised to see that Tom has managed it so well by efficiently focusing on small sets of data and candidates and producing complex features in multiple steps to amplify the power.  

Both Jack and Markus computed the feature probability distributions of place_ids.  There is an intriguing technique employed by Markus which essentially add the place_id predictions into training set to increase the power for future predictions.  I was planning on doing this as one of my next tricks to try.  I’m very pleased to see that Markus has already implemented it effectively and this helped him in getting to the second place! 

I’m very impressed by Jack taking meticulous care of feature transformation and probability calibration.  One technique particularly interesting to me is that he staggered the data in order to predict the future probability of place_ids, something that I did not think of.

There’s a lot more valuable experience in this competition I haven’t elaborated here.   My appreciation goes to all of you who shared your brilliant ideas!
