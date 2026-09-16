# 14th Place Solution: The Almost Golden Defenders

Competition: avito-demand-prediction
Rank: #14
Source: https://www.kaggle.com/c/avito-demand-prediction/discussion/60059

This competition was very overwhelming. I’ve never seen a Kaggle competition before where we’ve had to intelligently deal with images, text, geospatial information, unlabeled additional data, numeric data, categorical data, and a low-key time series problem to boot! Not to mention that we still don’t even know what the dependent variable is and a few of the independent variables remain a mystery as well.

Congratulations to everyone who got a gold medal -- especially the soloists. In my opinion, anyone who got a solo gold in this competition deserves two gold medals. I have no idea how you’d even manage something like that given everything this competition has to offer.

We’re really excited that we pushed way farther than we ever thought would be possible in the final weeks, but are really astonished by the caliber of the competition that we faced. I’ve never seen such a competitive final week. ...If only there were a few more gold medal slots, as there certainly were quite a few deserving teams.

Naturally, spending most of the competition in the gold medal zone only to lose out in the end by a smidge is a humbling experience. It leads us to second guess everything we did and wonder where the extra few points could have come from that could have made the difference.
 
Anyways, I’ll stop complaining and try to make up for our second guessing with an instructive write-up.


The Stacker
-----------

I’m too lazy to make a pretty diagram, but Sijun was not too lazy, [so you can read more detail here](https://www.kaggle.com/c/avito-demand-prediction/discussion/60059#350998). The original plan was to stop at Level 3 with just a single LightGBM, but the competitiveness of the competition forced us to keep going to level 4. Here's some brief discussion, though see Sijun's additional analysis.


**Level 4**

We created a four-level model, using a special Lasso model to aggregate together submodels. The two key features of this Lasso model was that the weights were constrained to be only positive and the model was trained separately for each `parent_category_name`. We found that `parent_category_name` introduced a good degree of variability in the errors and capturing this by building models separately was good for a very small (we’re talking ~0.00002) boost in score compared to training just one Lasso overall. Creating this insane Level 4 stacking system was key to boosting us from 0.2149 LB to 0.2148 LB on the public leaderboard.


**Level 3**

On the Level 3, we had 3x LightGBMs trained using (1) the L2 models and a few key features (parent_category_name, price, 20-dimensional SVD of text embeddings and a few image features), (2) all the L2 models and all the features used in L1, and (3) the same as the first model but trained with a Poisson objective. We also trained an MLP, a Lasso, and the aforementioned grouped Lasso by `parent_category_name` on all the L2 models (but no other features). Each of these models were also lazily bagged with some previous versions of the model that did not have some of our final models, for a total of 13 L3 models (7 LightGBM, 4 MLP, 2 Lasso). The LightGBM stackers each got ~0.2149 on the public LB and the Lasso ones each got ~0.2153. This was a large jump from Level 2 and Level 1.


**Level 2**

There was a small bump between Level 1 and Level 3 where we added all of our Ridge models to our LightGBM combined with all of our features. We trained this with both normal (regression) and Poisson objectives. We also lazily bagged this model for a total of four copies (3 regression,1 Poisson) by adding in previous copies of the model that didn’t have some of our final features. Our best Level 2 model scored 0.2173 on the public LB.


**Level 1**

We trained a ton of features into LightGBMs and varied the kind of encoding for the categoricals (either one hot encoding, LightGBM’s built-in target encoding, or Bayesian target encoding), varied the objective (regression and Poisson), and did some lazy bagging (using prior copies of the model that didn’t have all the features) for a total of 19 LightGBMs. Our best LightGBM scored 0.2183 on the public LB.

On the NN front, we trained three CNNs with FastText and two RNNs with Attention and pooling. Our best NN we 0.2181 on the public LB. [You can see how our best NN looked here.](https://www.kaggle.com/c/avito-demand-prediction/discussion/59917) We also diversified our Level 3 blend by adding two NNs trained with multiclass and one on binary objectives, which provided a good boost.

We also trained a large number of Ridge models that were varied on training just on TFIDF for the title, just on TFIDF for the description, just on character-level TFIDF for the description, and on all text (concat of all TFIDFs). We then trained these models individually for each parent_category_name, cat_bin (described below), and parent_category_name - region interaction. We also trained several Ridges on all the data and included interaction terms in one. All told, there were 17 Ridge models. All of these Ridges (but no other L1 models) formed the basis of Level 2.

Lastly, we trained three FM models, one using Wordbatch and two using TFFM. These didn’t end up outperforming our best Ridge models.


Features
-----------

Our feature engineering work was very exhaustive and tended toward a kitchen sink. In the beginning, we would diligently add each feature into the model one-by-one and keep it if it improved the first fold. Toward the end of the competition, we ran out of time for that, so we would just dump tons of features in at once. The one-by-one approach is nice because frequently there were features that looked good but ended up making the model worse.

We added a lot of features, so we’ll be posting separately sometime next week about everything we did.


Code Sharing
----------------

We’re still cleaning the code and getting everything together, and hope to fully share all our code sometime next week.


Until Next Time (Very Soon)
---------------------------------

See you guys more next week. I look forward to continuing to digest all the lessons learned and plotting my steps toward a real gold medal. We have some unfinished business. :)

![missed it by that much][1]


  [1]: https://i.imgur.com/Qj8kqVY.jpg
