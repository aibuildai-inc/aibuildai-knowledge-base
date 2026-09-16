# 1st place solution

Competition: playground-series-s4e11
Rank: #1
Source: https://www.kaggle.com/c/playground-series-s4e11/discussion/549160

Well, this is a nice and totally unexpected surprise! Congratulations to everyone who survived the shake-up, and thanks to @optimistix and @tilii7 for their support and encouragement last month, which kept me motivated to win the competition of this month.

I kept [my promise](https://www.kaggle.com/competitions/playground-series-s4e10/discussion/543772#3033958), @optimistix. Funny enough, my winning solution came from an [early experiment](https://www.kaggle.com/code/ravaghi/mental-health-prediction-1st-place-solution) I ran during the second week of the competition, which, honestly, I didn’t think had a chance of winning. Over the month, I trained many different models (69 in total) with various data pipelines and configurations, and tried many different methods to ensemble them. However, as it turned out, fewer models and a simpler ensembling approach worked better this time around. I ended up using 24 models (still a lot, though!).

The reason I doubted this solution would work was because the CV score was much higher than I expected (0.94173), and the public LB score was lower than many of my other submissions (0.94284). My other selected submission, which I spent a lot more time on, had a CV score of 0.94150 and a public LB score of 0.94285. Despite this, I couldn’t find any errors in my code, so I decided to trust the CV score and use this solution as one of my submissions.

My approach this month was similar to last month. I didn't do any feature engineering and did not drop any features, despite the temptation to remove the `Name` column. For the modeling part, I trained XGBoost and three variations of LightGBM on four different data pipelines, with the original dataset being included in two of the pipelines. I also used the OOF files from my [public notebook](https://www.kaggle.com/code/ravaghi/s04e11-mental-health-prediction-ensemble). Additionally, I trained two AutoGluon models, one with and one without the original dataset. These two models had the highest CV scores and were probably the two most important models in my ensemble. The scores for each of my models, and my AutoGluon ensemble can be seen in the figure below.

After training all the models and collecting their OOF files, I let AutoGluon handle the ensembling, ensuring to define the CV strategy myself to avoid leakage. It worked really well last month, so I decided to try it again, and it didn’t disappoint! I also experimented with ensembling the OOF files with hill climbing, Ridge, Logistic Regression, and a combination of Ridge and Logistic Regression, but the CV scores for these methods didn’t even come close to what I achieved with AutoGluon.


