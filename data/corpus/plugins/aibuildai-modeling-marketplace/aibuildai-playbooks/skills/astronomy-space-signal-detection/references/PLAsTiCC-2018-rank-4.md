# 4th Place Solution with Github Repo

Competition: PLAsTiCC-2018
Rank: #4
Source: https://www.kaggle.com/c/PLAsTiCC-2018/discussion/75011

First of all, congrats to the prize winners and all medal winners. During the competition, I have used the name Sisyphus because Kaggle Leaderboard was normally a daily routine for me to go up and down. But in this competition, I was the most stable Sisyphus ever till the last 3 days, staying on my comfortable 5th place. Then I saw the post from CPMP about their single model scoring 0.750 and I gave up because my blend was barely scoring that much. In the weekend, I went more experimental with this recklessness and it made me 3rd. Then I landed in 4th position in private LB.


My final score is a blend of LGB, NN and several stacking models. I will summarize my solution shortly. Later I will edit them with more details most probably. I will put **(G)** on the things that you can find on my Github repo. For now, my repo doesn't have the full solution but I will complete it soon:
https://github.com/aerdem4/kaggle-plasticc


**What didn't work for me?**


**Autoencoders**: Tried to encode the light curves. Generated vectors didn't help neither as features nor for clustering.


**Data augmentation using test set**: Tried to augment the data with the samples from the test set using the pseudo-labels. Improved CV, worsened LB.


**Remove background effect**: It seems background subtraction was noising the light curves a bit. Tried to normalize them as if they all have black background, couldn't find a way to do it.

**What kinda worked?**


**Probing class99**: I assumed that class99 are similar to certain classes. By probing, I ended up with the information that it is not similar to 15, 64, 67, 88, 90 classes predicted by my model. So I gave different constant to class99 depending on my prediction's highest class.


**Adversarial Validation &amp; Weighting**: Since train/test sets were obviously very different, I tried Adversarial Validation. Using all the features, it was very easy to distinguish train and test, so it didn't work for me. Therefore, I only used hostgal_photoz and ddf for weighting the samples. With these sample weights, my score improved around 0.02 and my CV was always higher (worse) than the ones reported on the forum. **(G)**


**What really worked?**


**Ratio features**: I guess most of you did this. Instead of using raw features from each passband, I used their ratio over all passbands.


**Stacking**: Even a simple Logistic Regression model on top of my LGB model's predictions (confusion matrix) improved my score around 0.04. **(G)** Later, I did more stacking.


**Hostgal_specz model**: I trained a model to predict hostgal specz using training set+ test set with hostgal_specz. Then used this model's predictions as a feature.


**Using normal values and log transformed values together on Neural Net**: Having both gives you the opportunity to do all four operations between the features (+, -, / *) because you can write log(xy) as log(x) + log(y). I guess hostgal_photoz was interacting with most of the features that I have.


**Bazin**: This is a light curve fit method that can be found on: https://github.com/COINtoolbox/ActSNClass/blob/master/examples/1_fit_LC/fit_lc_parametric.py I have changed it significantly and it gave me very big improvement. **(G)**


**Log Ensemble**: Instead of averaging predictions, I have averaged the logarithm of the predictions because in the end we try to regress the log values. This worked better for me. **(G)**
