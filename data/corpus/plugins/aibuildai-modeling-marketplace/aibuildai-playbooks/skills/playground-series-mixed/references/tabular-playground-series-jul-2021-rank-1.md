# Holy cow, I ranked 1st! Solution/Discussion

Competition: tabular-playground-series-jul-2021
Rank: #1
Source: https://www.kaggle.com/c/tabular-playground-series-jul-2021/discussion/256486

Hi everyone!

I had loads of fun playing and reading many of your discussions and codes! Below is a description of what I used and some of my thoughts. I learned a lot thanks to all of you! However, I need to publicly thank Alexander Ryzhkov @alexryzhkov for his very helpful discussions on Pseudo-Labeling and for publicly posting one of his notebooks. He deserves credit!


1. **Feature engineering:** Weirdly enough, I ended up not including any new variables beyond what was provided (+ the leaked data). I tried adding a wide range of variables, one at a time (dummies for months, weekdays, hours, etc. combinations of the weather variables, several interactions, lags, etc. etc.) but each of them worsened my score. This may have prevented overfitting the private dataset and hence may have helped my final rank, as pointed out by @jonaspalucibarbosa (https://www.kaggle.com/c/tabular-playground-series-jul-2021/discussion/256321). All of these extra variables were of course helpful prior to the leakage; post-leakage they didn't seem to help. 


2. **Missing values:** I estimated the missing (-200) values in the leaked data by using a two-stage estimation. @alexryzhkov post in the April competition was really helpful (https://www.kaggle.com/c/tabular-playground-series-apr-2021/discussion/231738). In the second step I trained the model using a longer sample: the training data +  leaked data + the imputed missing values. It was also particularly helpful to include the targets as features, e.g. use benzene and nitrogen to predict carbon monoxide. 


3. **Model:** It was very helpful to use 5 gradient boosted tree models, each with a different seed. I averaged the predictions of the 5 models. I used the same models in each stage; I didn't have time to try using a different model for the second stage.


4. Finally, averaging my output with Alexander Ryzhkov's LightAutoML  (https://www.kaggle.com/alexryzhkov/tps-lightautoml-baseline-with-pseudolabels) improved my final rank from 2 to 1. 

Something I'm still thinking about: For the time period the training data was covering, there seemed to be a difference between the targets provided by Kaggle and the targets in the leaked data (a difference beyond the -200 values). The difference averaged zero over time and it appeared to be correlated with the sensors. I tried to improve my score by modeling this difference (I commented some of these lines in the code), but I didn't manage to improve it despite the correlation with the sensors. 

Thanks all again! I would appreciate any comments or questions!
