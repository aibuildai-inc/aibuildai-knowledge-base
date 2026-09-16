# 6th place solution

Competition: happy-whale-and-dolphin
Rank: #6
Source: https://www.kaggle.com/c/happy-whale-and-dolphin/discussion/319829

Congrats to all the winners.
Thanks to Kaggle and the hosting team for an interesting competition.
I look forward to the third one:)

Here is my solution summary.

# Dataset
I used the full body and back fin data created by Jan. And I also used the results of training the detector using Jan's annotations. there were two different boxes for each fullbody / backfin.
I also used data with a slightly larger box. As a result, a fairly diverse set of data was used as material for the ensemble. And full images were also used as material for the ensemble too.
Image sizes: 512 ~ 896.

# validation
Since I started working with tensorflow late and was short on time, I took the approach of training with all trains and checking scores in LB.
Development without oof was quite difficult, but this time I thought it would not be a problem because the shake seemed to be quite small.

# Model
### tensorflow
All models are connected to dolg and arcface.
Dynamic margins were equally accurate with or without. Both were used.
- efficientnet v1: 5 / 6 / 7 / l2
- efficientnet v2: l / xl
- convnext: l / xl
Since I started using tensorflow in April, I ended up using the augmentation and hyperparameters as they are in the public notebook.

### pytorch
All models are connected to arcface.(without dolg, without dynamic margins)
- convnext :xl
- efficientnet: l2
- swintransformer: large384 (image size was 768)
I used a fairly heavy augmentation.

# Inference
I compared the similarity of the concated feature map between train and test.
The dimension of the final feature map exceeded 20,000.
Different thresholds were used to determine new individual id for each species.
no pp.

# iterative pseudo labeling
By using pseudo labeling, I can see not only the train but also the similarity to the confident test set. This is why pseudo labeling is important in this competition. So by repeating pseudo labeling multiple times, I was able to improve the score little by little.

# cat cafe
Since this was an 'animal competition', working in a cat cafe greatly improved my score.
[cat.png]
