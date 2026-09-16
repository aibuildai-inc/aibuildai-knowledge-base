# #9 Solution with CODE - Team BIG HEAD: Training model bonanza, TTA and L2 stacking.

Competition: rsna-intracranial-hemorrhage-detection
Rank: #9
Source: https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/discussion/117211

**CODE UPDATE**

Code is here: http://github.com/antorsae/rsna-intracranial-hemorrhage-detection-team-bighead

**SOLUTION OVERVIEW**

Our solution consists of pretty weak models (CV 0.07x in most of them) using L2 stacking (5 folds) trained with both _xgboost_ and _catboost_ and ensembled via averaging. 

We trained ~50 models (10 architectures/losses * 5 folds) in total.

The following table summarizes the architectures, folds, and GPUs to train each model:




**Fastai v1 3-slice networks: standard window and loss**

Architectures not highlighted (first five) were implemented using fastai v1 taking 3 consecutive slices (512x512) of a study and feeding them to the vanilla architecture with a fully connected head that outputs 6*3 = 18 logits. Training is done for 15 epochs using 1-cycle-policy. Batch size is allocated dynamically maximizing GPU memory usage. We use random rotations and flips as augmentation.

Loss function is the weighted average of the 3 slices giving more importance to the center slice:
```
W_LOSS = 0.1
GENERAL_WEIGHTS  = FloatTensor([2., 1., 1., 1., 1., 1.])
general_weights_3slices = torch.cat([GENERAL_WEIGHTS * W_LOSS, GENERAL_WEIGHTS, GENERAL_WEIGHTS * W_LOSS])

def weighted_loss(pred:Tensor,targ:Tensor)-&gt;Tensor:
    return F.binary_cross_entropy_with_logits(pred, targ.float(), general_weights_3slices.to(device=pred.device))
```

**Fastai v2 3-slice networks: subdural window and subdural focused loss**

We decided to use fastai v2 primarily b/c augmentations are done in GPU and a few of the computers we have had CPU bottlenecks doing augmentations, no longer the case with fastai v2.

Architectures highlighted in red we implemented as above with the following differences:
- Fastai v2 was used: much of a learning process and still has rough edges (some of them we realized after stage 1 finished and we could NOT change code). 
- Window centered at 100 and width of 254 (to take advantage of the range of `uint8`)
- Loss weighted on subdural more (10x) than other types:
```
SUBDURAL_WEIGHTS = FloatTensor([.8, .4, .4, .4, .4, 4.])
subdural_weights_3slices = torch.cat([SUBDURAL_WEIGHTS * W_LOSS, SUBDURAL_WEIGHTS, SUBDURAL_WEIGHTS * W_LOSS])

def subdural_loss(pred:Tensor,targ:Tensor)-&gt;Tensor:
    return F.binary_cross_entropy_with_logits(pred, targ.float(), subdural_weights_3slices.to(device=pred.device))
```

**Input to L2 Models**

Once models are trained we run OOF predictions using TTA with 10 repetitions. And we use the mean and std of those 10 TTA predictions for each architecture as input to both _xgboost_ and _catboost_, both for the central and surrounding slices.

Two L2 models are trained: _xgboost_ and _catboost_, and then simply averaged. One submission we did with the fastai v1 models only (they finished sooner) and the other using both.

**Things we would have done differently**

- Class-aware sampling (balance dataset)
- Pseudo-label training
- Fastai v2 head is different than v1 for vision models, the v1 head works better.
- We used a pretty high _eps_ for _Adam_ optimizer in v2, defaults (in v1) work better.
- Learnable window
- TTA with zoom, crops and cut-out
- Add extra channel with distance to center (similar to coord-conv but just radius to center) to make network location-aware.
- L2 model using lightgbm too and averaging 5 folds of L2 (we trained with 4 folds and hence we did not use all training set for L2)

...and the mandatory meme as a tribute to our team name:

