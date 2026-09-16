# 3rd place solution

Competition: g-research-crypto-forecasting
Rank: #3
Source: https://www.kaggle.com/c/g-research-crypto-forecasting/discussion/323703

First of all, I would like to thank G-Research, kaggle and all competitors. I have learned a lot through this interesting competition. It was very difficult for me (actually, it was the third machine learning challenge in my life) and without the knowledge shared by all the kagglers, I would not have been able to handle it.

Also, since this is a competition with a strong element of luck, it was only by sheer luck that I was able to finish in third place, but I am glad that my rank was somewhat stable in the last few updates.

In coding, I referred to the knowledge of kagglers. I am especially grateful to @jagofc for providing [local api](https://www.kaggle.com/code/jagofc/local-api/). Without @jagofc's local api, I would not have been able to successfully submit even once.

Here is my solution. If you have any suggestions for improvement or anything else you think I should know, I'd be glad to hear from you!

The link to the notebook is as follows;
- training: https://www.kaggle.com/code/sugghi/training-3rd-place-solution
- inference: https://www.kaggle.com/code/sugghi/inference-3rd-place-solution


### Overview of my model
The characteristics of my model are as follows
- Only 'Close' is used.
- The model is trained for each coin using a common set of features for all the coins.
- The difference between the change of each currency and the change of all currencies is provided as features.
- Single model of LightGBM (7-fold CV)

Considering the definition of the forecasting target in this competition, I felt it was necessary to prepare features with information about the entire market. I also thought that some currencies might be affected by the movements of other currencies, so I made it possible to refer to information about other currencies as well. Since I thought that memory and inference time would become more demanding with this kind of processing, I reduced the amount of data to be used. Specifically, I considered 'Close', which is used to calculate the target, to be the most important, so I decided to use only it. Even so, the ensemble could not be performed because of the limited inference time (and lack of coding skill).

For CV, I used [EmbargoCV](https://www.kaggle.com/code/nrcjea001/lgbm-embargocv-weightedpearson-lagtarget/) by @nrcjea001.


### About the data used for training
The start date of train data differed greatly among currencies. Since my model deals with the average of the changes of each currency, I considered it undesirable for the existing currencies to differ significantly between the training period and the evaluation period. As I expected that all currencies would have few missing values during the evaluation period of the competition, I decided not to use all of the train data, but to use the data from the period when there were enough currencies present.

The selection of the starting date was done by looking at the CV scores. However, this was a mistake in hindsight, since it meant that I was comparing CV scores across different data.

Also, each currency had several long and short blank periods. I attempted forward fill to prevent missing data as a result of rolling. On the other hand, I thought that forward fill for the entire period might cause a decline in data quality when there is a long blank period, so I set a limit on forward fill. In the evaluation phase, the code was designed to have forward fill without a limit, but I thought this would not be a problem since there are no long blank periods in the evaluation phase.


### Feature engineering
Since the value of cryptocurrencies is increasing, I tried to make sure to pick up the magnitude of the change independent of the evaluation period.

For 'Close', I prepared two features for multiple lag periods: the log of the ratio of the current value to the average during the period, and the log of the ratio of the current value to the value a certain period ago. For these, I took the average for all currencies (Due to missing data, no weighted averaging was performed). In addition, the difference between each currency and the average of all currencies was also prepared as a feature. As a result, this feature seems to have worked well.


### Dealing with time limit
The most difficult part for me was the time limit. When inference is performed, it should be sufficient to generate only one row of features, but my programming skills did not allow me to do this well, so I gave up on this and generated features for all data. So, I tried to avoid using pandas as much as possible to speed up the process, and managed to finish the inference within 9 hours. To be honest, I was quite worried about the timeout at the final update.


### What I would have worked on if I had more time
- Speed up feature generation to save time and perform ensemble.
- Learning with less missing data (external data?)
- Parameter optimization
The first successful submission was a week before the end of the competition, so I have not been able to optimize much.



I would like to continue to improve so that I can achieve results in competitions where the element of luck is small. See you at the next competition!
