# Solution | Private 4th

Competition: imet-2019-fgvc6
Rank: #4
Source: https://www.kaggle.com/c/imet-2019-fgvc6/discussion/94817#latest-550074

Congrats to all the winners.
Thanks to Kaggle and hosting team for an interesting competition.

Here is my solution summary.

### Pure Classification Model
Because LB scores of our classification models are similar or worse with many other competitors, I briefly describe the configurations.

-	**Dataset**: 10 folds CV with Multilabel Iterative Stratification
-	**Models**: se-resnext101, senet154. After last convolution layer, one more fc layer is inserted.
-	**Augmentation**: random resized crop, horizontal flip.
-	**Loss**: class balanced focal loss. But it didn’t improve score compared to the pure focal loss.
-	**Optimization**: AdamW, 
   - learning rate 0.00025, weight decay 0.01 for se-resnext101.
   - learning rate 0.0001, weight decay 0.05 for senet154
-	**Learning rate schedule**: CosineAnnealingLR. T_max 15, eta_min 0.00001
-	Input size 320x320. Batch size 24

With above configurations, the LB score of single fold se-resnext101 with hflip TTA was 0.618.


### Tag Relevance Prediction

We’ve tried several methods, but all were failed. So we’ve decided to improve prediction methods. 
We’ve found the method based on the [paper](http://class.inrialpes.fr/pub/guillaumin-iccv09b.pdf) improves LB score significantly. 

The detailed are the following:

For the training set, the presence probability of label *L* is defined as *1 – epsilon* if the example has label L, otherwise *epsilon*. The label presence probabilities of test example are the weighted sum over the nearest K train examples.


The weights are determined based on the distances between test and train examples. As a distance metric, 1 – cosine similarity between train and test example is used. The output of the last convolution layer is used as embedding features.

The weights of training example j for an test example i are defined as:

&gt; \\(\pi_{i,j}=exp(-d * distance(i,j))/\sum exp(-d * distance(i,j')))\\)

The probability of class *w* for test image *i* is:

&gt; \\( p(y_{i,w}=1)=\sum \pi_{i,j} * p(y_{j,w})\\)

The parameter d and K has been chosen based on the validation score.

With this method, the LB score of single fold se-resnext101 with hflip TTA was 0.634.
The average of two predictions is LB 0.650.

**Minor improvement**

We’ve found the images that have no close images have a low recall. So, we choose thresholds according to the distance from the nearest train image. If it has a higher distance, a lower threshold is assigned. Using this method, LB was improved +0.001~0.002.

Our final submission is an ensemble of 4 folds se-resnext101 and 6 folds senet154. 

Thanks.
