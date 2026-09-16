# 7th place solution

Competition: neurips-2023-machine-unlearning
Rank: #8
Source: https://www.kaggle.com/c/neurips-2023-machine-unlearning/discussion/459095

Firstly, I would like to express my sincere gratitude to the Kaggle platform and its dedicated team for their invaluable support and assistance. Their efforts are truly commendable. I feel fortunate to have secured the eighth position in this competition, and the experience has opened a new door to a fascinating field of study. I want to extend my appreciation to the other participating teams whose selfless sharing of knowledge is one of the reasons for the rapid growth and development of machine learning.

The code for our solution can be found at: [noise injection unlearning](https://www.kaggle.com/code/sunkroos/noise-injection-unlearning-8th-place-solution)

## **Overview:**
Our team designed a simple yet effective unlearning strategy that only requires the use of the retrain set. The solution comprises three main stages:

(1) Resetting the parameters of the fully connected (FC) layer and fine-tuning the entire model.
> By resetting the FC layer parameters, we can alter the distribution of the final output while preserving the features learned by the network. This effectively reduces the overfitting effect of the model on the forget set.

(2) Randomly selecting N layers from the network, adding noise to specific layers in an additive manner, and fine-tuning these layers.
> Adding noise helps the network 'forget' the information it has learned, and the randomness of the noise and layer selection contributes to enhancing the model's diversity. It is noteworthy that model diversity significantly affects the final score, as a single model distribution makes membership inference attacks (MIA) more likely to succeed.

(3) Fine-tuning all network layers.
> The primary goal of this step is to restore the accuracy lost in the network due to 'forgetting' information.

## **Tricks attempted but did not yield improvement:**

- Training with class weights.
- Changing additive noise to multiplicative noise, but no significant improvement observed.
- Calculating the importance of parameters in the retrain set and forget set based on model gradient statistics and adding more noise to parameters deemed more important in the forget set.
- Using label smoothing or focal loss to reduce overfitting/increase model generalization.

## **Unexplored methods due to personal constraints:**

*Due to limited resources, some potentially helpful methods were not explored.*

- Increasing the distance between original model parameters and current model parameters during finetuning.
- Using multiple unlearning strategies for ensemble learning in 512 models.

## **Conclusion:**
In conclusion, this competition has been a tremendous learning experience for me, providing insights into a completely new domain over the course of approximately two months. I am fortunate to have achieved a reasonably good result, and I hope that my modest contributions can be of some benefit to the field. I would like to reiterate my gratitude to all participating teams for their generous sharing. While this competition may not have resulted in medals, the ideas and thinking sparked from discussions serve as solid stepping stones along the path of machine learning development.
