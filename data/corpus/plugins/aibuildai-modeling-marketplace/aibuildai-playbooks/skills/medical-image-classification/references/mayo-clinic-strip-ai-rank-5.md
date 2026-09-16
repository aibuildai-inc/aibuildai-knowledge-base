# It was not all randomness ! (maybe) - 5th Place Secret Sauce

Competition: mayo-clinic-strip-ai
Rank: #5
Source: https://www.kaggle.com/c/mayo-clinic-strip-ai/discussion/357877

Although public LB was random (20 samples is not enough), you could actually get fairly consistent results.

Here are a few ideas, I will work on a more detailed write-up tomorrow :

1) Optimize the AUC, then you will know whether your model discriminates or not
My models achieve AUCs around 0.67, which is quite low but not random.

2) Understand the metric : the log loss heavily penalizes mistakes when the model is confident and the reward for getting a correct guess in comparison is much lower. 
Since our models are not so good, you want to stay on the sweet spot where your loss does not get heavily increased because of the mistakes your model makes. 

I did so by scaling predictions to the [0.15, 0.85] range, and then clipping them to [0.25, 0.75]. This was tweaked on CV, my best private achieved a 0.64 CV.

Sure it's only 0.05 lower than random predictions but that's probably close to the lowest you could get with the provided data :)
