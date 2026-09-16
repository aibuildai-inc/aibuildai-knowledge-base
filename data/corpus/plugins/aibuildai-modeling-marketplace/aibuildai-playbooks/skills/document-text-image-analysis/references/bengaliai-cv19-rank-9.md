# 9th Solution

Competition: bengaliai-cv19
Rank: #9
Source: https://www.kaggle.com/c/bengaliai-cv19/discussion/135985

**Model** 
My model backbone is simply based on seresnext50, using CV of randomly split 4 folds for training, no stratification. However, i have two extra loss heads which I found quite helpful to regularize the model - 1292-grapheme head CE loss and a Arcface loss of the grapheme head. 

However, the 1292 head is on top of the combined 3 tasks heads with binary mapping - 

 ```self.logit_to_1292 = nn.Linear(168+11+7, 1292, bias=None)```
 
```out_1292 = self.logit_to_1292(out_3tasks)```

where I initialize the `self.logit_to_1292` layer with a frozen binary weight that has dimension of (168+11+7, 1292), where it correctly maps each three combination of (grapheme_root, vowel_diacritic, consonant_diacritic) to its corresponding grapheme. I don't train this layer, however.

During back-propagation, I also used a very low weighting for the 3 tasks head and arcface head (0.02), but have a very high weight for grapheme head (1), which is quite counter-intuitive. 


**Augmentation**

I used 75% of CutMix, sample-wise. Also I gradually add in grid mask as training goes on, but I used 'grid-mix' instead - so instead of removing those grids, I randomly shuffle them within each image, so the per image stats won't change comparing to if you crop out those grids from image.  I found this slightly more helpful. 

For sample-wise cut mix (this is the most helpful data aug) - each image will randomly draw a binary cutmix mask during pytorch's data loading function, and I only shuffle and pair up the images within each batch during training, using this pre-computed mask. 

I also add in 5% of augmentation proposed by XingJian Lyu. 

Overall cutmix is the most useful part. 


**Self-training with external dataset 
**
I would say this is the part that helps my model much in generalizing to the test set. I only discovered the external dataset of ekush disclosed on the external data disclosure thread on the last week of the competition. 

Basically I downloaded and resize those 300K unlabeled images, and used my best trained 4 folds model to generate soft targets for the grapheme (1292) and 3 tasks heads (168+11+7). Then combined with original training data, I re-train my above model in 4 folds, with soft targets as labels. In each fold, I combine the entire 300K external data plus the training folds to train each model, and do this 4 times. 

This approach is similar to the idea of noisy students self-training paper shared earlier on in the forum, where I generated soft targets without noise, and retrain the model with noise. The student and teacher models are exactly same as above. 

The key thing is how you do sampling with those unlabeled dataset - the one that works best is first to generate hard labels for graphemes, and then combined with our train dataset, to use Heng's balanced sampler to do sampling based on all graphemes. Other sampling strategies simply don't work. 

This finally pushes my model from 994 to 996 on Local, and on LB to improved from 985 to 990, where ensembling of the 4 folds further boosts it to my current score of 916 (all based on public score) 

**Things didn't work for me**

OHEM
Efficientnet (slightly worse than seresnext50)
senet152
dropblock
attentiondrop
shake-drop
intensive-augmentation or autoaugment
using flip or 180 rotate as addition identities (strategies that worked best for humpback competition)
