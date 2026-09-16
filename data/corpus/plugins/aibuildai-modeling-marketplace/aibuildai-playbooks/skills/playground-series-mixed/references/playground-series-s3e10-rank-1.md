# 1st place solution

Competition: playground-series-s3e10
Rank: #1
Source: https://www.kaggle.com/c/playground-series-s3e10/discussion/396345

My solution is [here](https://www.kaggle.com/code/seascape/how-to-detect-pulsars-with-gam-1st-place). I used GAM, however, with two additional features created with XGB and LASSO. That is, the final model is ensemble, although I think a little less "classic". Importantly, these additional features were not so much important here. The GAM model was my first approach, and without these features I was getting only slightly worse results. 

I chose this model for a few reasons:

* We have very few variables that are already high-level features. Feature engineering is basically impossible, it's just a matter of finding the right relationship.

* Boosting is based on trees, which, in my opinion, are not able to find "real" relationships. Because real world relationships are rather continuous in nature --- and decision trees only approximate it. Of course, they are great and you can approximate almost anything (e.g., very high-level interactions), but the world does not work in that way.

* GAM has built-in regularization; it selects parameters by performing internal crossvalidation. This makes it very convenient, and I could freely add interactions and features created from the XGB and LASSO model.
