# 15th place solution (0.047) --- Close but no Cigar!

Competition: rsna-intracranial-hemorrhage-detection
Rank: #15
Source: https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/discussion/117277

**I see that many of the top 10 have already posted their solutions, but we would still like to share ours!**

**One reason is** that the starting point of our [best] solution really was the [public kernel](https://www.kaggle.com/akensert/inceptionv3-prev-resnet50-keras-baseline-model). What was added, thanks to my team mates, was albumentation's augmentation (which would be put in `_read()`), as well as bigger input size and more epochs of training (these are pretty obvious). There were also some alterations in the learning late schedules, but that's pretty much it! 

Then as the "next level", we would use [about] a dozen of models (including B2-B6, Xception and InceptionResnetV2 in addition to InceptionV3) to ensemble. ResNe(X)ts didn't work for us (or rather, not for me), so they weren't included in this particular submission.

Below is the method to squeeze out as much as we could from the individually trained models that we had (scores of ~0.068-0.074), which would eventually get us to 0.058 (public LB).

Four (or three) levels of averages (wisdom of the crowds at work!):

*We had 3 separate ensembles: (I) mix of architectures (90/10 train/val split), (II) 6-CV B4, and (III) 6-CV InceptionResNetV2.*

**For each ensemble in ensembles:**
**(1).** All models' individual predictions are a weighted average of each epoch's prediction (a.k.a. snapshot predictions). The weights are optimized via scipy.optimize.minimize, resulting in 1 prediction matrix (M x N\_classes) per model.
**(2).** Now each model's predictions are also averaged (but not optimized weights, because we don't have a validation set here). Now we are at ~0.065 on public LB.
**(3).** Every possible validation prediction data point (from **(1)**) is used to train two sequence models (2-layered LSTM and 2-layered GRU; see figure (credits to my teammate @ratthachat :-))), which are then used to "correct"/predict the test set predictions. Target Y would be the true labels (N\_classes=6) and the input would be the sequence of predictions for each StudyID's slices (sorted from low ImagePosition3 to high ImagePosition3) and including the floating point value of ImagePosition3 (N\_slices=60, N\_features=6+1). This step really boosts our log loss: we now advanced from 0.065 -&gt; 0.059 just like that! Each Sequence model (RNN) was trained for 35 epochs, and like **(1)**, we used scipy.optimize.minimize to compute a weighted average of all 35 epochs' predictions (so yet again using the "average snapshot prediction method"; seems to work really well for us)
**end For :-)**



We now had 3 ensemble predictions (3 submission files), which were simply averaged and submitted to Kaggle!

**So to summarize**, we used pretty standard models with lots of averages and good post processing. 

Unfortunately (for us) it didn't go all the way to the top 10, but we're still extremely grateful for this competition! You know, creating friendships and all that!

Also, good job everyone! :-)
