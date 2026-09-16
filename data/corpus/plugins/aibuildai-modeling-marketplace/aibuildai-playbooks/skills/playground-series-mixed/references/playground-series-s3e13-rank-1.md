# Surprisingly #1, Public LB: 0.37196 | Private LB: 0.53179

Competition: playground-series-s3e13
Rank: #1
Source: https://www.kaggle.com/c/playground-series-s3e13/discussion/406433

First of all, I would like to thank Kaggle for organising this Playground Series. It was such a nice experience for a competition beginner like me.

To be honest, my main aim in joining this competition was to learn how to use autoencoders within an ensemble model. Practically, it was heavily inspired by [this legendary thread](https://www.kaggle.com/competitions/porto-seguro-safe-driver-prediction/discussion/44629). The second aim is to get the merchandise :D.

Therefore, I understand that I did not put a lot of effort into pre-processing the data, and could say that I was very lucky in this competition.

I joined this competition a bit late, so I was only able to produce 4 types of models. No feature engineering, all features being used were scaled using a standard scaler.

| Model Name | Public LB | Private LB | Notes |
| --- | --- | --- | --- |
| LightGBM | 0.31677 | 0.41337 | Nothing fancy, just a simple LightGBM model with default parameters |
| Neural Network | 0.33995 | 0.44078 | Simple NN using 64-64relu-32relu-11softmax as the layers. The first 64 is for the input layer = num of features on the dataset |
| Autoencoder | 0.37196 | 0.46052 | The autoencoder uses bottleneck architecture 64-64relu-32relu-16relu-32relu-64linear. Take the encoder part (up until the 16relu), freeze it, and add 16relu-11softmax on top of it |
| Ensemble | 0.35871 | **0.53179** | This is a simple averaging ensemble model from the previous three models. Below this table is the explanation. |


Each of the models would be able to generate each class' probability. I thought that maybe by averaging each class' probability on each model (for example, averaging the probability of "Malaria" from LightGBM, Neural Network, and Autoencoder), I can somewhat make an educated guess. 

I assumed that if something is very convincing for most models (let's say, Neural Network and Autoencoder are convinced that the outcome is Malaria), but not that convincing for the other (maybe Dengue for LightGBM), I would still choose the majority vote here.

Therefore, what I did was to average the probabilities and get the top 3 probabilities after averaging them.

My notebook is a mess and clearly lacks any explanation, so I am sorry that I could not post the notebook publicly. 

I was thinking of doing stacking as well but I haven't managed to get my code working, so I stopped here.

I am very open to discussion and would like to hear any thoughts from you. Have a great week ahead!

Also, I would like to thank @belati and @mpwolke . You guys being very active in this competition has encouraged me to keep trying!
