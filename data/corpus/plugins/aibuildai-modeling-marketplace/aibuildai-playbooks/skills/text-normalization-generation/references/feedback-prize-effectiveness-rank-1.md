# Team Hydrogen: Efficiency Prize 1st Place

Competition: feedback-prize-effectiveness
Rank: #1
Source: https://www.kaggle.com/c/feedback-prize-effectiveness/discussion/347537

Thanks a lot to the hosts and Kaggle for hosting a separate efficiency track in this competition. Also thank you @ryanholbrook for answering our questions and addressing technical issues for this new format.

### Summary

Our most efficient model is a single deberta-v3-large that got a 0.557 score on the Private LB in 5 minutes and 40 seconds. So, this model alone would get a top-3 position on the Private LB with such a quick submission. Here you can find a notebook with our most efficient submission: https://www.kaggle.com/code/philippsinger/team-hydrogen-efficiency-prize-1st-place

### Model design

The model architecture follows the same principles we’ve described in our [main solution](https://www.kaggle.com/competitions/feedback-prize-effectiveness/discussion/347536) (essay group model). However, the main difference comes from the way this model has been trained. We were fortunate to figure this approach out two days before the competition deadline:

1. We generate pseudo labels data for the previous Feedback competition by using our large ensemble with 2nd level models. In the main solution we used 3 rounds of pseudo labeling, but for this model we used 4th round
2. We also generate out-of-fold pseudo labels for our given train data
3. Now we combine these 2 datasets with soft pseudo labels together and train a single new model without any original labels

It allows us to distill the knowledge from the large ensemble to a single model. And it is doing that pretty well, only losing 3 points vs our best large ensemble submission.

Additionally to a deberta-v3-large model, we have a couple of extra 2nd level models to further improve the quality of the predictions. More details about our 2nd level models are available in the [main solution description](https://www.kaggle.com/competitions/feedback-prize-effectiveness/discussion/347536). Note though that these extra 2nd level models only bring one additional point here.

### Inference optimization

During the competition we always had some plans to optimize the speed of the model inference, but have never found time/priority for it. So, our final kernel is a simple PyTorch model with a cleaned inference loop. One thing that allowed us to save another 40 seconds was to pre-tokenize essays in advance and sort them by the sequence length. Such an approach in combination with dynamic padding is 40 seconds faster compared to just sorting the essays by the char length. 

We also tried Deberta-small and Deberta-base models, both of them also work really well and you would only lose a couple more points in accuracy, while even reducing the runtime by some minutes more. For example, a single base model runs in less than two minutes!
