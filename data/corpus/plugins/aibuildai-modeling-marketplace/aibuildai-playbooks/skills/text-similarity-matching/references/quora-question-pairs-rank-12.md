# Solution #12 overview

Competition: quora-question-pairs
Rank: #12
Source: https://www.kaggle.com/c/quora-question-pairs/discussion/34342

Here is an overview of our solution as seen by me.  I hope my fellow teammates will correct/augment what I say here.  First of all, let me thank them for great ideas and code, hard work, and great team spirit, including when we were losing ground on LB.  I also want to thank the numerous kagglers that shared useful content. I name few of them below but there are way more to thank.  And thanks to kaggle team and the organisers for this very interesting dataset.  Of course, we all have quesitons about the sampling strategy used to construct that dataset, and any feedback from Quora team would be much appreciated.

This competition mean lots of learning for the four of us.  I personally learned a lot about NLP as it was a first time. I didn’t know what word embeddings were when I started for instance.  Fortunately, the official external data thread plus posts by various people contained great starter lists.  

I will focus mostly on the xgb/lgb side of our solution, and stacking, and will let our DL experts, kenchen and yifanxie describe the NNs side if they feel inclined to do so.

We used 4 magic features families, 3 of them having been shared publicly one way of another.

Graph node features.  Looking at the graph where questions are nodes and rows are edges, we computed node degree (Jared’s frequency), connected component size, biconnected component sizes, average neighbor degree, etc

Graph edge features.  Using same graph, computed features based on common neighbors, eg number of common neighbors, number of common neighbors divided by sum of number of neighbors for both questions, etc

Temporal pattern.  We did not spent much time on that, but we have one feature that is the abs difference of each question rank in order of appearance.  Ranks are computed for each side (question1, and question2) independently.  We observed that the order of questions has an effect but did not model it.

Transitivity of is duplicate.  If q1 is duplicate of q2 and q2 is a duplicate of q3 then q1 and q3 are most probably duplicates.  We say most probably because this is not 100% true on the train dataset.  A variant is: i If q1 is duplicate of q2 and q2 is not a duplicate of q3 then q1 and q3 are most probably not duplicates.  One way to capture this is to to say that if q1 and q2 have similar properties with respect to their common neighbors, then they are probably duplicates.  We therefore compute, for each level 1 prediction p, and for each row (q1, q2):
the series of predictions p(q1,q) and the series of predictions p(q2,q) for all q common neighbours of q1 and q2.  We then compute distances between these series, and add it as a feature for level 2.  We improved this in the last day of the competition by using the target value instead of train predictions in an out of fold manner.  

While the first three feature families may be artifacts of the sampling method used for constructing the dataset (aka leaks), the last one is not.  It is exploiting a mathematical property of the is_duplicate relation.

We also used a lot of NLP features, starting from ones shared by @anokas and @abishek, and also drawn from previous competition winners, especialy @ChenglongChen github repo.  We computed features on raw text, text cleaned in various ways, and stemmed/lemmatized.  We used various word vectors: pretrained Word2vec, Glove, Fasttext, but also computed ones using gensim dbow2vec, and lsi.  For each embedding we computed distances similar to those shared by abishek, and also 10 component pca of the abs difference of question vectors.  Using pca has the advantage of not basing any feature on the order in which the vector components are presented.

Most of the NLP features are used in their pairwise form.  For instance, if l1 = len(q1) and l2 = len(q2) are the length of each question in a row, we use min(l1, l2) and max(l1,l2) as features instead of l1 and l2.  We also took the ration of the smallest to largest in some cases.  Goal is to not have features that depend on the order in which questions appear in a row.  Another way is to duplicate rows in each fold: if (q1, q2) is a row, then add the row (q2, q1).  This avoids overfitting on very popular questions whatever the other question they appear with.

We used class weight or output scaling to improve LB  When using output scaling there is a trap that may explain why stacking failed for many teams.  If you input rescaled level 1 test predictions to your level 2 learning, then you must also rescale the level1 out of fold train predictions.  In general, whatever you do to compute your test predictions must be applied to the train predictions.  

We generated about 100 level 1 models over the course of the competition, mostly using xgboost and lightgbm on the one end, and NNs on the other end.  In the last couple of weeks we also added xgb gblinear, random forest (H2O and sklearn), extra tree classifiers, logistic regression, and few others, to add variety to the ensemble.  Our best submitted model is a xgb with 600+ features, with LB 0.131x.  We may have built better models after that but we didn’t submit them individually.

We built 2 ensemble, one made of models that did not use any of the 4 magic feature families, and one where these features were allowed.  For both ensemble we used xgb, lgb, logistic regression, and NNs.  Transitivity post processing of level1 predictions was used for xgb/lgb. Level 3 classifier was a logistic regression.  We tried to apply transitivity again at level 3 but that led to overfit.  Stacking yield about 0.01 LB improvement overall.

Work was split that way between us:
Cpmp: feature engineering, xgb, stacking 
Ss: feature engineering, lgb, additional algorithms run
KenChen: NNs, feature engineering, 
Yifanxie: NNs, feature engineering, xgb and other algorithms runs

We also learned about teaming as we never worked together before.  It has been a very tiring, but very rewarding experience.  Our 0.003 improvement the last day of the competition was the best ending we could dream of.
