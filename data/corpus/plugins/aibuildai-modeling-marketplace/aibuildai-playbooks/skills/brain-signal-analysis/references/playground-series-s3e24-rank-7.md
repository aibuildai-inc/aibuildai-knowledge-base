# #7 Private LB and #2 Public LB solution

Competition: playground-series-s3e24
Rank: #7
Source: https://www.kaggle.com/c/playground-series-s3e24/discussion/455271

First, I congratulate all the winners and all the participants in the Binary Prediction of Smoker Status using the Bio-Signal competition.

- Initially, I used the model with ***XGBoost*** and optimized with ***optuna*** over a many iterations and no feature addition ([public score: 0.87392](https://www.kaggle.com/code/sarunpm/easy-binary-classification-smoking-xgboost-optuna#Optuna-Optimizer))
- Then the ***pseudo label technique*** seen in the public notebook of @cv13j0 was added and gave an additional boost in the public score of 0.87901 (landed me in top 100). Tried higher degree of leaking of test dataset to train dataset up to 85 % [Marginal improvement in public score]
- Then, comes the public notebooks of @zhukovoleksiy and @arunklenin which looks almost similar. But the notebook of @arunklenin was unique because of  ***the added tons of derived features*** and ***averaging of prediction probabilities of kaggler having top public scores*** ([Average Method](https://www.kaggle.com/code/sarunpm/0-88083-in-12-sec-blending-of-topper-s-notebooks?scriptVersionId=150176729)). Incorporating these features improves the score a bit in top 20 public score (0.88116)
- I also observed that the features like, **hemoglobin, weight, height, Gtp, serum creatinine and dental caries** are important features and added features like ***hemoglobin x hemoglobin, hemoglobin x height, hemoglobin x weight, weight x height, hemoglobin x Gtp, hemoglobin x serum creatinine***, and ***many more features in combination with hemoglobin*** which landed me in **top 2** public score (0.88126) which I improved to `0.88136` in last day by changing the **SEED** from `42` to `43`. (*Changing seed has this much impact !!??!!*)
- Finally after the results my final standing is in **7<sup>th</sup>** position with private score of `0.87926`. But, I have some past submission with lower public score have better private score than what is showing in the leaderboard. ***Why??*** [Why]

Anyway, I am happy because of the achievement and I would specially thank the authors, namely,  @rukenmissonnier, @paddykb, @arunklenin, @cv13j0, @alexryzhkov for sharing their notebook whose submission was used to making my weighed submission file. I also appreciate @ravi20076, @oscarm524, @yaaangzhou and @armanzhalgasbayev,  as I learned many new ideas from them and already forked their public notebooks on the present competitions.

Thank you all
