# #16 place solution and some questions about it.

Competition: allstate-claims-severity
Rank: #16
Source: https://www.kaggle.com/c/allstate-claims-severity/discussion/26430

Hi guys and girls,

First off all, I’d like to congrats the winners and thank all the people who contributed in the forum, Tilii, Vladimir Iglovikov, MT, Faron … to name just a few.

I’m new to data science and this is my first competition, as much this is solution sharing it’s also about validating it. So feel free to challenge it, point out any unnecessary/wrong steps and how I could have improved it.

I used 4 different algorithms: XGB, LGBM, Keras, and sklearn GBR. All models are 10 fold cv, with two main transformations: power(loss+0, 0.26) and log(loss+200), but the first transformation was better.

**XGB**: I used fair objective with constant=1.5, Best model: CV=1124.79, LB=1105.25. (10 models)

**Keras**: same architecture that the one shared in the forum with default MAE. Best model: CV= 1130.59, LB=??. (7 models)

**LGBM**: best model: CV= 1128.08, LB= ??. (6 models)

**SKlearn GBR**: best model: CV=1136.37, LB=??. (2 models).

I had 25 first level models that I stacked using the same 4 algorithms. I used the two transformations above, but also with no transformation of loss for sklearn GBR, which significantly helped my score (I didn’t get the time to experiment with the other ones). I also added three extra features: the mean of all 25 models, cont14 and cont7 from training set.

**Best CV scores are:**
XGB= 1117.37, bagged by changing the model’s seed.
Keras= 1118.44, with 2 layers: 200(0.35), 80(0.1) and a constraint on the weights (maxnorm=0.5).
Sklearn GBR= 1116.65, bagged dozens of times by changing the model’s seed.

My final score is an optimized weighted average of the first 25 models and 14 second level models. I used the scipy minimize package, with the BFGS method. The score on the oob predictions is **1115.66, LB=1099.48 and 1111.38** on private LB, which gave me my current rank.

**These are some of my questions**:

Is adding the mean, and features from the training set common practice in stacking?

The BFGS method gives negative weights, is this normal, how can we interpret this? In all cases, it outperformed the other method that just give positive weights (SLSQP), in both oob predictions, public and private LB.

My weighted average includes first and second level models, is this common practice?

My 2nd level stacked models performed worse than the optimized weighted average, is it because all models are similar in performance and that I should have incorporates some “worse” ones?

Thanks to kaggle and to kagglers, and I to the pleasure of racing with you again in the LB :)
