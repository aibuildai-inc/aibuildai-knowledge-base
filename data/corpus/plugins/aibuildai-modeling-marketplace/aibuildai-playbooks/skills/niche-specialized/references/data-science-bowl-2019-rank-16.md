# 16th place solution

Competition: data-science-bowl-2019
Rank: #16
Source: https://www.kaggle.com/c/data-science-bowl-2019/discussion/127331

Thanks to everybody for a great competition. Congratulations to the prize winners. I really enjoyed this one even if dealing with QWK was infuriating at times.

The outline of my solution is as follows:

Feature engineering was fun. I used a lgb model to assess new features working pretty much on minimising the MSE. In the end I developed a lot of features then instigated a cull using CV to reduce the number to a final 158 features.

The top features were those based on the previous performances in the title that we wanted to predict as well as those in other assessment type activities. Counting occurrences of certain words in previous activities (like “misses”, “rounds”) also proved helpful if split by the title in which they occurred. Features based on the amount of game time spent on each event code in each title also produced some good features. (For example, event code 4070 in activity 12 was particularly helpful.)

Having settled on a feature set, I then used this in a standard lgb model using MSE as the objective, ran it through a NN as well as augmenting the data with the unused test set assessments for a third model. They all produced similar results in CV. An ensemble of these three models produced my final model. In common with many, I used a repeated random selection of the 3614 installation ids, truncated, to estimate a QWK. For the third model above I used a classification objective. I then optimised each class probability estimate using the truncated CV setup. This produced an optimal output of 1.62p1+1.74p2+2.64p3. (A standard output of 1p1+2p2+3p3 scored well but not quite optimally.) 

Blending and thresholding were tricky but the truncated CV setup seems to work ok to optimise QWK. I was least sure about this step though it appears to have been reasonably accurate with regards to the private lb. I pretty much ignored the public lb scores but was still pleased to survive the shake-up!
