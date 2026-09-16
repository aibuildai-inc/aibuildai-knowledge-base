# 1st Place Solution - CatBoost All The Way Down

Competition: playground-series-s4e10
Rank: #1
Source: https://www.kaggle.com/c/playground-series-s4e10/discussion/543725

Hey Kagglers! I used to be pretty active in these playground competitions, but after the December 2023 competition I took a break from Kaggle. On a whim I decided to start working on this one about 10 days ago, and it's been as much of a thrill as it ever was. Getting 1st place was a surprise, to be sure, but a welcome one!


**Cross-Validation**
I'm sure you've heard this before, but setting up a robust cross-validation scheme for evaluating the performance of your predictions is **VERY** important to doing well in these competitions. I see lots of questions from folks on what kind of feature engineering to do, or how to best ensemble models, impute data, engineer features, etc. For a vast majority of these questions, there's no single answer that is universally true for any dataset. The only way to find out what works for a particular dataset is to try various options and see what performs the best, and that's where cross-validation comes in. In these playground competitions, the data is usually split 60-40 between train and test set, and 20% of the test set is used for the public leaderboard. That means that a CV score measures your performance on **60%** of the entire dataset, whereas the public leaderboard measures your performance on only **8%**, making cross-validation performance a much more reliable indicator of progress than public leaderboard performance. All of the decisions made below were based on optimizing my cross-validation performance.

**Data Preprocessing**
Shoutout to various member of the community for the tip to treat the numerical features as categorical. What I found most effective was to maintain both the numeric feature and a categorical copy of it. I didn't do any other feature engineering, as my experience from past playground competitions has usually been that feature engineering is of little use. I did include the original dataset.

**Modelling**

My general approach here is the same as the one I used last competition. For each of XGBoost, LightGBM, and CatBoost, I used Optuna to find 10 different sets of 'optimal hyperparameters' and averaged their predictions to get an overall prediction for each. Shoutout to @omidbaghchehsaraei's post [here](https://www.kaggle.com/competitions/playground-series-s4e10/discussion/539963) for the tip to use large max_bin values. I also added a Neural Network that was heavily inspired from @paddykb's notebook [here](https://www.kaggle.com/code/paddykb/ps-s4e10-no-keras-no-loan-cv-0-963). The performance of each of these models is as follows:

| Model| CV Score| Public LB| Private LB|
| ------------- |-------------| -----|------------- |
|LightGBM       | .96811 | .97005| .96637 |
|XGBoost    | .96767 | .96989| .96540 |
|CatBoost | .96972 | .97299 |.96865 |
|NN| .96678 | .97088 | .96577 |

<br>
What I think might have been my secret sauce was that for each of these model predictions, I trained a CatBoost model using the initial model predictions as a baseline. An example of how to do this can be found [here](https://catboost.ai/en/docs/concepts/python-usages-examples#baseline). I'm not sure exactly what inspired me to do this, perhaps it was from seeing how amazingly well CatBoost performed on this data, but to my surprise CatBoost was able to significantly improve the performance of each of these model predictions, even the ones that were originally generated using CatBoost. The performance of these CatBoost-improved models are as follows:

| Initial Model | CV Score| Public LB| Private LB|
| ------------- |-------------| -----|------------- |
|LightGBM       | .96856 | .97048| .96713 |
|XGBoost    | .96815 | .97024| .96611 |
|CatBoost | .96997 | .97334 |.96903 |
|NN| .96732 | .97117 | .96667 |

<br>
I find it impressive that the CatBoost model that used CatBoost predictions as a baseline would have been enough for 3rd place. CatBoost was the king for this comp! The final step was a Neural Network to stack these 4 predictions together. This squeezed out the extra last bit of performance needed to bring the solution to the top. 

| CV Score| Public LB| Private LB|
|-------------| -----|------------- |
| .97059 | .97344| .96938 |
