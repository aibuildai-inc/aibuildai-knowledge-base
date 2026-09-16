# 25-th place solution: CosFace + ProtoNets

Competition: humpback-whale-identification
Rank: #25
Source: https://www.kaggle.com/c/humpback-whale-identification/discussion/82409#latest-488027

Here I would describe my part of the solution: CosFace approch.
The ProtoNets and Ensebling would be described by @daisukelab

**1. Preprocessing**
Fist of all, I use BB from @radek (thank you!). I just did train model on his annotated data and that's all, I did not invest time for this part of competiton. In later stage I also used updated BB from @radek (I call them v2), but the difference in final results was very small.

**2. Model and Data-Loading**
My final model was se-resnext101 (I also try se154 but it did not work nice). In fact, my model and augmentation was exactly the same like in this kernel: https://www.kaggle.com/stalkermustang/pytorch-pretraiedmodels-se-resnext101-baseline
 Other stuff which I tried:
 -  CutOut -&gt; fail
 -  MixUp -&gt; fail
 - OverSample -&gt; fail
 - Cluster-Based Sampling: here I try to sample batch so that there were similar classes (based on cosine similarity) -&gt; fail, overfitting

**3. Loss Function**
In the past, I was developing the Face-Recognition system and the main purpose of approaching the whale problem was testing such technology on whales:) So in general I was testing ArcFace and CosFace. Both of them works pretty good but CosFace was slightly better. For CosFace I use very high margin 0.6 (in original paper it was 0.35). 
In fact, this step take me the longest time (3 weeks), where I was trying:

 - BatchNorm vs LayerNorm before L2 normalization: LayerNorm better
 - AlphaDropout vs DropOut: Alpha better but with no influence in debugging model (resnet50) so I did not use both of them, what now I think was one of the biggest mistake I made
 - CosFace vs ArcFace vs SphereFace: CosFace with m=0.6 was clear winner (I also the the idea of the CosFace the most)

**4. Optimalization**
Here I use AdamW (with fixed weight-decay). I also try OneCycle but it was not working (I think that   code was wrong). In general I train the model by 30 epochs, so it was pretty quick.

 - New-whale
I did not use 'new-whale' for training. In sumbission I just want to have ~27% of 'new_whale'
I have two approches for this problem which did not work:
 - Each new-whale as different class: In general it was ok, the accuracy on validation set was just 0.2% less. But it does not work well on LB.
 - Use second loss-function (KL-Divergence) which would force the outout distribution afer SoftMax of 'new-whale' to be uniform (so exactly the same probability for each class). It also work fine for validation set, but not in LB. 

Look like I would need more time for this approach (especially second one), because I really like it:)

The final models was se101resnext trained on:
- gray and 448x448
- gray and 256x748
- rgb and 448x448
- rgb and 256x748

In general, the approch was pretty simple and work moderate. But look like @pudae had the similar idea, so I'm now looking into his approach :)

The ProtoNet and Ensemble part would be explained by @daisukelab.

Code: https://github.com/melgor/kaggle-whale-tail
This is minimal code for training single model and create sumbission without 'new-whale'
