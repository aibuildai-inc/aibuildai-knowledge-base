# 10th Place Solution

Competition: trends-assessment-prediction
Rank: #10
Source: https://www.kaggle.com/c/trends-assessment-prediction/discussion/163212

Thanks to the organizers for this exciting competition and also congratulations to all the winners.

Here is a brief description of my solution.

.png?generation=1593597555606133&amp;alt=media)

- My final submission (public: 0.15665, private: 0.15741) is an average of several variations of the prediction above.
- The blending part of the pipeline above uses not only OOF predictions but predictions on training sets (Gaussian noise is added to reduce overfitting). This improved public LB score 0.1577 -&gt; 0.1569.
- I used only ridge regressor as the 2nd level model. The parameter alpha and the scales of features were optimized with Optuna.
