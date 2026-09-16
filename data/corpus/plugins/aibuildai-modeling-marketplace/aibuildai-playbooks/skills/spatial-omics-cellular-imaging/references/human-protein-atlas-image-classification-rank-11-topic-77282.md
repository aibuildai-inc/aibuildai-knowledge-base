# 11th Place Solution

Competition: human-protein-atlas-image-classification
Rank: #11
Source: https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/77282

Congratulations to each of the kagglers, it was a very interesting game, and thanks to each of the selfless kagglers on the discussion.
In the meantime, I really appreciate the hard work of each of my teammates, and without their experiments and their GPUs, I don't think we can get the gold medal.

##The following is our experiment:
**[Update]**

[The code of our solution ][2]

### The first stage experiments 
&gt; We used the external HPA data in Gray format(512 size)

**Models**

 - res18 (batchsize=64)
 - res34 (batchsize=32)
 - bninception (batchsize=32)
 - inceptionv3 (batchsize=32)
 - xception (batchsize=24, P40-24G) 
 - Se-resnext50(batchsize=24, P40-24G)

**Data Augumentation**

 - train
&gt;  Add/Multiply/Crop/Affine/Filplr/Filpub/

 - 12 TTA

**Optimizer**

 - NAdam with different LR for different layers 

**Loss Function**

- bce

**Threshold**
&gt; We tried the search threshold，but it was not work, so we finally had no idea and chose 0.205 as threshold.

**Result**

- The best score from single model with 5 fold was 0.597(public)

### The second stage experiments 
&gt; we changed the format external HPA data (you can find this in discussion)


**Ensemble**
we used the first and second stage model to ensemble, and it should get 0.62+ score(public)

### The third stage experiments
&gt; [My teammate, shisu's method][1]


  [1]: https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/77289
  [2]: https://github.com/Gary-Deeplearning/Human_Protein
