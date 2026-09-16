# 11th place solution

Competition: LANL-Earthquake-Prediction
Rank: #11
Source: https://www.kaggle.com/c/LANL-Earthquake-Prediction/discussion/94363#latest-548737

Many congratulations to the winners, The Zoo, you truly stood out on the private leaderboard! Also congratulations to all teams that survived the private earthquake or learned something during the competition. @Keita111 (+4024 to claim a gold with 2 submissions) is a great example that we should sometimes focus less on the public leaderboard even though we all know we should :-). 

Special thanks go to all forum contributors. People like @CPMPml, @mykper, @scirpus, @Abhishek and several others are what make Kaggle great and I am deeply grateful for your contributions.

Let’s start with what didn’t end up in the final submission. My efforts during this competition were mostly wasted on modeling the gaps (every 4095/4096 observations) in the data. The assumption was that if one could detect these gaps, it would be possible to order the test chunks, making the test prediction problem trivial. This model works perfectly on the validation data but it doesn't on the test data. I am certain that the test data is not contiguous as stated and is actually divided into 150,000 observation chunks with random gaps in between. It is a bit sad that the organizers ignored many good questions about the data distribution in the “Additional info” topic, requiring the competitors to make many assumptions. 
Side note: the gap analysis identified test chunks 1db8e8, 35a2d7, 35dd45, 395e0e, 62a403, 996c37, a35e7c, d1eee8 as the start of an earthquake cycle, adding another piece of information to believe that the test data really comes from the test part of P4677.

The second biggest chunk of my time was spent on unsupervised learning of features using the raw high frequency data with recurrent networks. The code was inspired by [David Tellez's](https://github.com/davidtellez/contrastive-predictive-coding) implementation of [Contrastive Predictive Coding](https://arxiv.org/pdf/1807.03748.pdf). The hope was that this would lead to robust features for training and testing. I embedded each chunk of 1,500 observations to a shared embedding of size 32 and defined the following loss heads:
- Predict if one embedding directly follows (with some small random gap) another embedding
- Predict if a pair of embeddings comes from the same data chunk of 150,000 observations
- Predict if an embedding comes from the train or test data
- Predict if a pair of embeddings comes from the same earthquake
- Predict Time To Failure (TTF)

The first three losses are trained for both the train and test data of which the third one is trained using [domain adversarial training](https://arxiv.org/pdf/1505.07818.pdf). The last two losses can only be trained for the train data. Even though I ended up not using any of these models, it became clear that it was very hard to distinguish if raw observations come from the same earthquake. That led me to redefine the learning objective in the final approach.


**Actual submission**
An alternative way to specify the learning objective is the following: predict the quantile TTF of an earthquake, meaning that the target is 0 at the beginning of an earthquake cycle and 1 for the last 150,000 observation chunk of each earthquake. Assuming that this model is trained to minimize the MAE of the quantile, what should each (1-prediction) be multiplied with to minimize the **MAE**? The median of [the earthquake lengths, repeated by the earthquake lengths]! For example, if the test data consisted of earthquake lengths 6, 3 and 2 this would result in median(6, 6, 6, 6, 6, 6, 3, 3, 3, 2, 2) = 6 * (1-quantile_pred) as the optimal predicted time to failure. The nice thing about this reformulation is that we can use the data from the P4677 experiment (I estimated the test median earthquake TTF to be 12) without having to break your head over what earthquake cycles to train on.

The models themselves use basic FFT and quantile features for each 150,000 observation chunk and 100 1,500 observation subchunks. This resulted in 40*101 = 4040 features which were fed into LightGBM (feature fraction of 0.003) and neural networks. The first final submission averages the LightGBM and neural network predictions starting from the last 6 complete earthquake cycles. The second submission uses all data except for the first incomplete earthquake cycle (given that the cycle length is unknown we can’t determine its quantile). A nice property of this approach is that the predictions don’t change much when training on different folds so in hindsight it would have been better to submit with different estimated median private test earthquake times.

I look forward to your feedback!
