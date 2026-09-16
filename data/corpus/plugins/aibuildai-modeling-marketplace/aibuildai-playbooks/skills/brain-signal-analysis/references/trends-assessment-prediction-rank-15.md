# 15th Place (basic tabular part)

Competition: trends-assessment-prediction
Rank: #15
Source: https://www.kaggle.com/c/trends-assessment-prediction/discussion/162903

Thanks to Kaggle and the host for putting on a really interesting and challenging competition. I was fortunate to have super talented teammates  @cpmpml @philippsinger @robikscube  and @fakeplastictrees.   It was a great experience and it is easy to see why they are so successful.  There are some things you can only learn by watching experts at work, so I'm grateful they teamed up with me.  

Congratulations to all the gold medal finishers!  I always learn so much from reading solutions.  Thanks for writing them up!

You should definitely read Rob's solution https://www.kaggle.com/c/trends-assessment-prediction/discussion/162749  He did some very interesting work on the 3D resnet. 

## Basic Tabular Model
 I'd like to thank  @aerdem4 for his very popular SVM notebook and @tunguz for the ensemble and KNN notebooks that I combined for the basic tabular model.  

I tried using OOF values to fillna but that improved my CV but did nothing to the LB.  One interesting thing I did find is that oof age was just as good predicting the other 4 targets as fnc features or  loading features.  

I got a 0.0002 LB improvement by predicting the 4 domain targets with oof age and then predicting the residuals with fnc and loading features.  

At the point, I teamed up with my fantastic teammates.  

##fMRI features
Thanks to @broach and his notebook https://www.kaggle.com/broach/trends-working-with-fmri-component-maps.  I used that as starting point and calculated summary stats for all of the regions of interests, which I fed into a lightgbm.  Our blends seemed to like this model for domain2_var2.   I tried a few different versions and if I had more time, I would have liked to have explored this area more.  I'm very interested to read the solutions from domain experts to see how they used this information.

##TSNE/UMap
Thanks @tunguz for the TSNE/UMAP notebooks.  I felt like there was something there, but I never quite could get to it.  I'm looking forward to reading how PCA was used.
