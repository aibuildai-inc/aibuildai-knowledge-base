# 4th Place Solution for the March Machine Learning Mania 2025 Competition

Competition: march-machine-learning-mania-2025
Rank: #4
Source: https://www.kaggle.com/c/march-machine-learning-mania-2025/discussion/572466

#  Title: 
4th Place Solution for the March Machine Learning Mania 2025 Competition


#  Context: 
Business context: https://www.kaggle.com/competitions/march-machine-learning-mania-2025/overview
Data context: https://www.kaggle.com/competitions/march-machine-learning-mania-2025/data


#  Overview of the Approach:
Logistic regression (win or loss) on the leaf nodes of Xgboost with Cauchy loss function on point differential. Features include all features of the original https://www.kaggle.com/code/raddar/vilnius-ncaa but I added Laplace Smoothed features based upon prior season matchup, away wins, last 14 day win ratio, and  uniquely stacked out of fold predictions. For the validation strategy, I only use personal intuition based upon the Brier score of the last year 2024 being left out.


#  Details of the Submission: 
I did not use the CV code listed because of time based leakage. I did not use the spline in the original code because of target leakage. I used the leave one season out only for model averaging instead of CV. I had models with "better" leaky local CV which I did not submit. No gambling was used. No manual manipulation was used. My other submission was Raddar's old Github published solution in R which won multiple times in the past. I did not use any external data and only utilized data published on Kaggle for official use in this competition. 


#  Sources: 
My code: https://www.kaggle.com/code/mikeskim/gold-medal-solution-mike-kim
This code should exactly reproduce my submission file except on matches including non-seeded teams. These are all set at 0.5. It would take too much RAM and time to produce a file with all possible matches including non-seeded teams within Kaggle Notebooks.
Based upon: https://www.kaggle.com/code/raddar/vilnius-ncaa
https://github.com/fakyras/ncaa_women_2018
