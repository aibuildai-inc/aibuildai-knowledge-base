# 10th Solution

Competition: imet-2019-fgvc6
Rank: #10
Source: https://www.kaggle.com/c/imet-2019-fgvc6/discussion/95311#latest-568748

First of all, I would like to thank FGVC6,CVPR2019 and Kaggle for hosting such an interesting competition. I would like to thank my teammates Jiwei Liu and Yuanhao Wu for their hard work and brilliant ideas. Special congrats to Yuanhao for earning GrandMaster title! 

We feel really lucky to get the last gold medal. We only scored ~0.615 on the team merger deadline, lots of efforts were done in the last week. 

**Dataset**: 5 fold CV with Multilabel Iterative Stratification
**Models**: seresnext50, seresnext101, pnasnet5large, inceptionv4 and senet154. 
**Augmentation**: 
**Train**: Resize, RandomSizedCrop, HorizontalFlip, Blur, GaussNoise, HueSaturationValue and CLAHE
**Test**: Only Resize
**Loss**: BCE loss for the 1st stage, F2 loss for the 2nd stage
**Optimizor**: Adam, LR=2e-4. 
**Learning rate schedule**: ReduceLROnPlateau(patience=0, ratio=0.1)
**Input size**: 331x331 for pnasnet5large, 384x384 for other models. 
**Batch size**: 128. 

We started the journey with seresnext50 using BCE loss. Small modifications were made to the network, our logit takes input from the last two layers rather than the last one layer. When we finetuned the entire network, we found that a significant boost in f2 score is achieved right after the LR reduces, so we decide to set patience to 0. Later, we found that if we only train the last two layers, we can achieve similar score compared to training the entire network. Hence, we decided to only train the last two layers and classifier, this saves us lots of time and allows us to apply big batch size without batch accumulation. Meanwhile, in another experiment, we found that training the best model achieved by BCE loss using F2 loss for 3-5 epoches could give another big boost (~0.01). This finalizes our solution: train the model with BCE loss until converge (~0.603 on LB for single fold of seresnext50) -&gt; switch loss to F2 for another 3-5 epoches (~0.612 on LB for single fold of seresnext50). 

Then it comes to the final week. We don’t have time for more experiments, so we just applied the above strategy to other models and keep generating 5-fold results. A summary of our model performance (most models only have CV scores, since we don’t have enough submissions for all the models in the last 2 days):

**Seresnext50**: 0.6078
**Seresnext101**: 0.6207 (0.646 on LB)
**Pnasnet5large**: 0.6066
**Inceptionv4**: 0.6030 
**Senet154**: 0.6212 (0.642 on LB)

Our ensemble method is a simple weighted average over the above models, it scores 0.658 on public LB. 

1. What we tried but not work:

2. All crop related augmentation methods failed. 

3. TTA does not work for us. 

4. Efficientnet only has a CV of 0.5956, maybe we have not use its potential fully. 

5. We tried batch accumulation, our test shows there around 0.001 loss in local CV score. We didn’t accumulate too many batches, perhaps this is the reason. Personally, I think there is a trade off between the big batch size and BN synchronization problem. 

Our solution is a little bit brute force compared to other teams. Congrats to all winners. Thanks for reading!
