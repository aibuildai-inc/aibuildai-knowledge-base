# [8th] Place Solution for the [Predict Calorie Expenditure] Competition

Competition: playground-series-s5e5
Rank: #8
Source: https://www.kaggle.com/c/playground-series-s5e5/discussion/582631

Despite my last submission being 16 days old, I was surprised to secure 8th place in the final standings. My solution utilized a weighted ensemble optimized via a Hill Climbing algorithm. The final ensemble combined five distinct test predictions: three TensorFlow models, one CatBoost model, and one XGBoost model.

My primary objective in this competition was to deepen my understanding of TensorFlow through hands-on practice. By focusing on tensor manipulations and library-specific architectures, I was able to build specialized models that contributed significantly to the diversity and overall performance of the final ensemble."

**Notebooks Ensemble of Predictions Summary**
Notebook # F_1027 is an XGBoost public notebook you can find [here](https://www.kaggle.com/code/jiaoyouzhang/calorie-only-xgboost). Thanks to @jiaoyouzhang 

Notebook #F_1023 -> CatBoost public notebook. Many thanks to @chrisk321 .You can find it  [here.](https://www.kaggle.com/code/chrisk321/lb-0-05696-ps5e5-solo-cb-model)

The rest below are all tensorflow from my private notebooks:
Notebook #F_1025, F_1021 and F_1024 

I did not bother optimizing or doing some unique stuff on the public notebooks. I just mixed this up with my Tensorflow models and call it a day. 

Logs from Hill Climbing algo:

`Current Best model:
F_1023
Models to add to the best model: 
Starting RMSE:  [0.058812503]
Iteration: 1, Model added: F_1025, Best weight: 0.41, Best RMSE: 0.05860217
Iteration: 2, Model added: F_1027, Best weight: 0.27, Best RMSE: 0.05848809
Iteration: 3, Model added: F_1021, Best weight: 0.12, Best RMSE: 0.05847134
Iteration: 4, Model added: F_1024, Best weight: 0.05, Best RMSE: 0.05846814
complete`
