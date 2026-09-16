# 43rd Place Solution for the Predict Calorie Expenditure Competition

Competition: playground-series-s5e5
Rank: #43
Source: https://www.kaggle.com/c/playground-series-s5e5/discussion/582806

This is the plan I followed.
* Analyze the data.
* Make a preliminary model without looking at other approaches or using any libraries. 
* Make a model w/ a good CV-score and send in two submissions.

After [analyzing the data](https://www.kaggle.com/code/jliangwork/p5e5-data-analysis/), I determined that 

"1) the variables most closely associated to `Calories` are `Duration`, `Heart_Rate`, and `Body_Temp`. 2) While the relationship between `Duration` and `Calories` seems to be positive and linear, the relationship between `Heart_Rate`/`Body_Temp` and `Calories` seems to be S-shaped” - quoted from my `preliminary model` notebook

I then made a [preliminary model](https://www.kaggle.com/code/jliangwork/p5e5-preliminary-model) using the following

**Assumption:** We can treat the effects of `Duration`, `Heart_Rate` , and `Body_Temp` on `Calories` as additive without interaction terms.

**Reason Behind Assumption:** I felt like it.

I trained this model through simple gradient descent. I used all the data points to calculate the gradient for each update. In retrospect, my preliminary model did not perform much better than a linear model trained on least-squares regression (after a `log1p` transformation of the target) as the CV results in my notebook show.

Next, I trained some models using standard libraries (e.g. `XGBoost`, `tensorflow`, etc.) and read the community discussion. From the community, I adopted the following insights.
* Feature engineering seems to be of limited use, although scaling other variables by `Duration` may be helpful.
* When creating voting rates, we can have different weights for different `Sex` and `Age` bins ( @harukikakinuma ) - I ultimately only had different weights for `Sex` and did not consider age.

My final ensemble (weighted voting) included 5 forest-type models, 4 trained through boosting algorithms and one trained with a vanilla random forest algorithm, and 2 neural network models trained using gradient descent. Here are details of the training.
* The learning parameters for the forest-type algorithms were tuned through grid search.
* The architectures and learning parameters for the neural networks were tuned through vibes.
* The voting weights were tuned through `optuna`. I originally wanted to use a grid search, since I don’t yet understand how `optuna` works, but my grid was too large.
* The algorithms were scored via overall RMLSE of the OOF predictions, rather than the average RMLSE over folds - see my post [HERE](https://www.kaggle.com/competitions/playground-series-s5e5/discussion/581196).
* My best CV score was around 0.5918. The unweighted average achieved a CV score was around 0.5919. I used 4 folds, mainly for time issues.
* I clipped the predictions of the neural network models to be in-range. No clipping is necessary for forest-type models to address in-range issues.

I did not discover anything interesting about the features, model architectures, or learning parameters not discussed elsewhere in the community.

My two submissions differed in the following manner.
* In Submission 1, I used the best `number of trees` for each forest-type model that I found while training.
* In Submission 2, I increased the `number of trees` for each forest-type model by 1/3 to account for the fact that my training sets in CV were 3/4 the size of the overall training set.

They received almost the exact same score on the private LB (0.05855 and 0.05854, respectively).

**Main Thing I Learned:** How I like to organize my workflow.

**Goals for Next Competition:** 
1. Make more home brewed models.
2. Interact w/ more community members.
