# 1st place solution

Competition: tabular-playground-series-jan-2021
Rank: #1
Source: https://www.kaggle.com/c/tabular-playground-series-jan-2021/discussion/216037

First of all, I want to say that I feel incredibly honored to win (first time for me in the „spotlight“ on kaggle 😏)! I really liked the dataset - even though it was labeled as "not much of a challenge" (in my opinion it was really challenging and fun and I learned a lot).

As already guessed, my solution is partly based on **DAE-transformed** data which I
used as input for quite heavily regularized **M**ulti**L**ayer **P**erceptrons (**MLP**s). 

I basically fed both, train and test data, into a first level Denoising Autoencoder and used it's weights as a new data set for a 2nd level NN (see picture below). 

The cool thing about a DAE is that if done correctly the model's weights will include a lot of feature information (no more feature engineering needed 🔥). And as it turns out DAEs could capture some information that helped stage two NNs.

In addition, there are, who would have thought it, some Lightgbm models. Both variants (DAE into MLP and lgbm) were trained in different setups and finally stacked with the help of a ridge regression model and simple averaging. I spent most of the time on DAE training/ validation. The lightgbm model I used is based on slightly adjusted @kailex params (💪).
My final and winning submission is a ridge stacked ensemble of multiple DAE-MLPs and some lightgbm models:
- local 10 fold cv = 0.69289
- public score = 0.69530
- private score **0.69381**

All my NNs stacked via averaging score:
- local 10 fold = 0.693773
- public score = 0.69620
- private score = **0.69472**

If you are curious about DAE-data. I created a custom dataset for testing. Feel free to use it. I'd love to hear your thoughts and findings! 

✔️[LINK TO DATASET](https://www.kaggle.com/springmanndaniel/dae-representation/)

I also did a little summary/ writeup about my solution.

✔️[LINK TO SOLUTION](https://www.kaggle.com/springmanndaniel/1st-place-turn-your-data-into-daeta)

Cheers,
Daniel
