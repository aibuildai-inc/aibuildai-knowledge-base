# Rank 2 approach - a century of component feature sets and deep ensemble

Competition: playground-series-s5e2
Rank: #2
Source: https://www.kaggle.com/c/playground-series-s5e2/discussion/565542

Hello all,

Thanks to Kaggle for the intriguing episode in the Playground series! This was such a different episode from the usual Playground episodes! Thanks to my fellow participants for such a healthy competition throughout the month.

My overall approach was a deep blend of boosted trees, neural networks and ridge model with deep feature engineering with the architecture as drawn below-



# Feature engineering
- As discussed in several public kernels and posts, this was the most important and crucial component of the competition.
- I used the column WeightCapacitykg as a float and its **string twin** as separate features. In my sample dataset, this column is labelled as **WeightCapacity**
- I prepared a total of 1600+ features across 9 datasets and stored them in a feature store to retrieve and use across the month. 
- I used OrdinalEncoder for all string/ category variables and then combined them into 1-2-3-4-5-6-7 gram combinations and stored them in separate datasets for easy retrieval
- I also used the idea of joining the original features from the kernel [here](https://www.kaggle.com/code/cdeotte/feature-engineering-with-rapids-lb-38-847) and used them with my engineered features in 2 separate datasets. I considered only bigrams and trigrams here as the number of features exploded a lot and resources were not sufficient to handle the volume of data generated.
- I used Colab TPU to prepare these features as I obtained a virtual machine with more than **200GB RAM** on my TPU. I stored all my component features in separate parquet files for easy retrieval and usage and used **Polars for subsequent feature retrieval** and usage. 
- I used TargetEncoder from CuML for all my encoding purposes and used `mean, median, count, nunique` as aggregators
- I have open-sourced a set of features for you to peruse [here](https://www.kaggle.com/datasets/ravi20076/playgrounds5e2featurestore).

# CV scheme
I used a 20-fold cross validation scheme for all my models, including the classifier to keep consistency across all single and blended models 

`cv = KFold(n_splits = 20, shuffle = True, random_state = 42)`

# Level-1 Model training
- I trained catboost, xgboost and lightgbm models as my layer-1 models on separate feature sets drawn from the gamut of features described above.
- Each model comprised of a separate feature set, with important features in common. Certain features like `WeightCapacity, WeightCapacitykg, Brand, Brand-Color-Size, Brand-Material-Size, Brand-Color-Material-Size`, etc. were almost always present, while other features were model specific. I identified top 50 important features and retained them in all my models, and varied the features otherwise in my component models.
- I designed a total of **65** boosted tree models with the below CV scores across single model solutions - 

| Model type |Number of single models designed  | CV score range | 
| --- | --- | --------- | 
| XGB  Regressor   |  21 | 38.6463 - 38.75856 |
| LGBM Regressor |  20 | 38.6471  - 38.66303 |
| Catboost Regressor|  21 | 38.6480  - 38.74479 |

<br>Additionally, I also designed separate boosted tree models with the Autoencoder as below-

| Model type |Number of single models designed  | CV score | 
| --- | --- | --------- | 
| XGB  Regressor   |  1 | 38.65556|
| LGBM Regressor |  1 |  38.65727|
| Catboost Regressor|  1 | 38.658758|

- Additionally, I also designed a simple dense NN **classifier** with integer targets for some diversity to the ensemble. This model was a poor choice for a single submission, but it added a needed diversity to the ensemble and created a minor gain upon blending. 

| Model type |Number of single models designed  | CV score | 
| --- | --- | --------- | 
| Dense NN classifier   |  1 | 38.891892|

# Public artefacts
- I used the public autoencoder model with a few adjustments and also executed the kernel [here](https://www.kaggle.com/code/cdeotte/feature-engineering-with-rapids-lb-38-847) and used them in my ensemble
- Thanks to the authors of these kernels!

# Level-2 Model training
- I had to blend these boosted tree models into a meaningful ensemble and thought of using a simple MLP as a stacker model. I used **Kaggle and Colab TPU** to train these NN models. One often uses CPU and GPU resources but procrastinates the 20-hour of TPUs available per week! Using these resources to good effect was key to a lot of experiments this month!
- I created a total of 35 stacker NN models with varying model OOF features with the CV range as below- 

| Model type |Number of single models designed  | CV score range | 
| --- | --- | --------- | 
| NN stacker   |  35| 38.63546 - 38.65008 |

# Level-3 Model training and post-processing
- This is the last layer in the model process, a simple ridge model that blends the results of the L1- boosted trees + L2 NN models, the public artefacts and the MLP classifier model for the submission
- My final submission contains a combination of **100 models** and has a **CV score of 38.62860836 and a Public LB score of 38.82326 and a private LB score of 38.62947** 
- I also rounded the predictions to the nearest integer value - this slightly downgraded the CV but improved the LB score. Since the CV was less optimal, I chose this as an alternative submission. This one scored slightly lower on the private leaderboard with **score of 38.63039**

# What did not work for me
- Sample weights 
- Appending the original data with the competition data 
- Variance and std-dev aggregators in features while performing target encoding
- Post-processing - I found a few repeated rows between the `train and test sets` and between the `train and extra training data` as well. Copying the targets across these common rows in my submission result did not help me at all
- Using any model other than Ridge in level-3

# My key takeaways from the assignment
- This was a GPU intensive assignment and I learnt how to manage my resources better with such a lot of GPU oriented training through the month
- I learnt the art of using TPUs for FE and model training here. TPUs work wonderfully with NNs and this is a fast way to iterate through multiple experiments quickly!
- I became better at feature encoding with emphasis on Target Encoding - this is a big gain for work-assignments as well!

# Training GPU stack

| Stage | GPU/ TPU usage  |
| --- | --- |
| Feature creation | Colab TPU  |
| XGboost | A6000 Ada + 128 GB RAM |
| LGBM  | A6000  + 128 GB RAM|
| Catboost| A6000Ada x 2  + 256 GB RAM|
| L2-NN| Colab TPU/ Kaggle TPU|
| Ridge| Local CPU |
| Model parameter tuning + feature experiments | Local GPU (3090) + 128GB RAM|

# Concluding remarks
Congrats to my fellow swag prize winners, best wishes to all the participants and happy learning to one and all!
Hope for the best in the upcoming Playground episodes and across Kaggle featured competitions as well!

Regards,
Ravi Ramakrishnan
