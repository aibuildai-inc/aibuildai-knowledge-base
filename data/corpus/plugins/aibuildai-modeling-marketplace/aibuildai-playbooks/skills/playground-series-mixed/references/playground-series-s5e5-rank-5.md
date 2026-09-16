# 5th Place Solution: Ensemble of 68 models

Competition: playground-series-s5e5
Rank: #5
Source: https://www.kaggle.com/c/playground-series-s5e5/discussion/582564

**Introdution:**
In this competition, the main challenge is unstable CV-LB correlation and FE is not working very well.
So I decided to bulid an ensemble and fully ignore public LB.

**Workflow:**
In the beginning, I focus FE with forward selection by brute-force column-wise aggregation.
But most models keeps at <20 features without improvements.
Then, I switched to ensemble.

**Hyperparameters(HPs):**
I don't use Optuna or Grid Search either. Instead, I save all the OOFs(and test predictions) of all tested HP combinations.
Since FE is not effective, I decided to train (too) many models with different HPs.

In the last week of the competition, I have trained
LGB, XGB, CatBoost(GPU), HGB, YDF, Random Forests, a total of more than 2000 tree models and 10 MLPs(GPU)

**Hill Climbing (HC):**
I like HC because it can perform model selection by forcing a model weight to exactly zero.
(Lasso can also do that but in general it is too hard to tune optimally.)
Then, I perform HC sequentially as a model selection tool until a smaller set of OOFs does not give better CV.
Note that this process is order-variant, I permuted the OOFs for thousands of times to get the final subset of 79 OOFs.

Then, I used `cvxpy` (quadratic programming) to perform HC again with the remaining 79 OOFs.
`cvxpy` gives deterministic result that should be order-invariant and generally better and faster than sequential HC.
I didn't use it in the beginning because it has very high memory usage.
It ends with 68 OOFs.


**Result:**
Final submissions
CV: 0.0588017, Public LB: 0.05671, Pivate LB: 0.05846 (`cvxpy` select 68 OOFs from 79 OOFs)
CV: 0.0588070, Public LB: 0.05671, Pivate LB: 0.05846 (Sequential HC)

Best unselected submission
CV: Forgotten, around 0.05885~0.05890, Public LB: 0.05692, Pivate LB: 0.05845

In the last few days, I also found that CV-LB correlation is much more stable with ensemble prediction than single model prediction. 
In my case, via HC, better ensemble CV means better LB.
The final 2 submissions are also my best CV & public LB.

Ensemble is the key in this competition, and NN is the key of the key.
I had trained > 2000 tree models and only 10 MLPs
However, in my final ensemble of 68 models, 7 of them are MLPs, they also contribute the most in my ensemble!

**Takeaways:**
1: Try different strategies in the beginning. 
I discovered a stable CV-LB correlation and power of NN in ensemble too late. So I don't have much time to try more NNs and more ensemble techniques.
2: Bulid a good diversified ensemble than combining strong single models.
3: Trust CV. Blending public works is fine, but if the weights are determined by luck/LB probing, then shake up is coming when CV-LB is unstable.

Since FE is not important in this competition, I don't provide the process of FE and model fitting, the actual code is just a standard routine of 5-Fold CV with a outer for loop of different HPs.
The usage of `cvxpy` and also my solution is provided [here](https://www.kaggle.com/code/alan1305/pgs505-5th-place-solution-cv-0-0588)
