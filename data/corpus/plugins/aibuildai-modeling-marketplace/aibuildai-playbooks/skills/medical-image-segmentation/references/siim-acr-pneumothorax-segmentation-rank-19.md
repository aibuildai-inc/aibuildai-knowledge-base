# 19th place solution

Competition: siim-acr-pneumothorax-segmentation
Rank: #19
Source: https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/discussion/107666

First of all, congratulations to the winners.
This 2 month long journey felt like eternity 
because at the end I have no improvement for quite long time,
but fortunately thank you for wonderful team member @ildoonet   joined
at the end, and shed some light with his hypercolumns Unet with efficient b7 backbone.

Architecture:

We used 5 Hypercolumns U Net models, with Resnet34 and Efficient B7(by @ildoonet ) backbone.

Model 1: 

4 folds of Hypercolumns U Net Resnet34 with bce+dice

I found that bce+dice loss surpass bce loss in terms of convergence speed and dice score.
And rather it seems extreme, we used 5 phases cyclicLR scheduled training and 
reduced the last layer's max learning rate at each phase(lr/2,lr/2,lr/4,lr/8,lr/16)
and took best dice score model as an ensemble candidate to minimize uncertainty comes from the updated stage2 dataset.

The best dice score cv results for stage2 retraining were (0.8560/0.84519/0.8367/0.8536)

model 2:

4 folds of same configuration with self attention

Self attentioned model get lower dice scores(0.842/ 0.8535/ 0.8504/ 0.8412) but 
There was an improvement when ensembling self attentioned model in stage 1.

model 3, 4, 5:

model 3,4,5 are 4 folds models of @ildoonet , Hypercolumns U Net models with Efficient U Net B7 backbone.
It reached its full performace faster compared with model1,2 so we used 3 phases cyclicLR scheduled training.
model3,4 has different learnin rate configuration 8e-3 and 1.6e-2.
The best dice score cv were (0.8452,0.8481,0.8445,0.8550) for model 3 , (-,0.8482, -, 0.8417) for model4(some logs was missed)


and model 5 trained by @ildoonet  's configuration, I believe he will explain on his own write up 
but single model can scored private LB score of 0.8536.


TTA :

We used only Horizontal Flip.


Postprocessing:

We removed noises with mean probability of the prediction for each cells.

First with fixed probability threshold(0.21), define a cell with its connectivity.
Then by its mean probability over 0.35 remains as pneumothorax, and below this probability discarded.

In stage1, this boosted my model 1's 4 folds ensembled model(mean dice score 0.84515, Phase4) at stage1 LB 0.8729
compared to similar dice score(mean 0.84485, Phase3) with thresholded noise removal at stage1 LB 0.8658.
(Unfortunately I lost my direct comparison when LB was reseted at stage2)

Scores : 
0.8568(Stage2), 0.8763(Stage1)
