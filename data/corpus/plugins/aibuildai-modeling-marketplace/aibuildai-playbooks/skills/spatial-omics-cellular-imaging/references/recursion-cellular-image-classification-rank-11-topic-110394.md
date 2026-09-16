# 11th place solution - AttentionHeads

Competition: recursion-cellular-image-classification
Rank: #11
Source: https://www.kaggle.com/c/recursion-cellular-image-classification/discussion/110394

# AttentionHeads
First of all, many thanks to my teammates, @dempton, @ddanevskyi, @orgunova and @cutlass90. Also our congratulations to @dempton for getting his Grandmaster badge.
So, here is our (pretty simple) approach.

# Model
* Ensemble of 2 **EfficientNets**: B0, B5
* **6 channel** input with first Conv layer replaced with 6 channel version
* **Batch Normalization** before first Conv layer
* We normalize each image using mean/std **computed on all images in experiment**

# Training
* Simplest approach possible: **3-fold split, stratified by cell type**, as well as by **visual appearance**: We made simple visualizations of experiments within cell type and tried to split similarly looking experiments into different folds, so to have all kinds of images in every fold.
* No smart sampling or special losses, just basic **Cross Entropy**, without label smoothing or metric learning
* **SGD** with [LookAhead optimizer](https://arxiv.org/abs/1907.08610) (a=0.5, k=5) and 4-step **gradient accumulation**
* [1cycle policy and LR range test](https://sgugger.github.io/the-1cycle-policy.html) to find best learning rate for training
* **Polyak averaging** (exponentially weighted version). When evaluating model we used exponentially weighted average of model parameters instead instead of last/best parameters, this does not give any performance improvements of final model, but has very pleasant **smoothing effect on metric/loss curves** and gives huge performance improvements in almost all training steps except for the very last, where it reaches same performance as model without averaging
[polyak averaging effect]
* Augmentations: randomly sample site and apply flip and transpose, as well as random **channel reweighting** (just multiply each channel by some positive values with restriction that they should sum to 6)
* **Progressive resize** during training: starting from random crops of size 224 we linearly scale crop size to 512. This gives **2x faster training** without drop in performance
* 1 round of **pseudo labeling**. We just selected top-K samples with most confident predictions, added them to each fold's train set and finetuned our models for several additional epochs

# Post-processing
* **LAP solver** within each experiment to assign classes to images
* **Softmax Temperature**. Just multiply logits by some positive number before taking softmax `(logits * t).softmax()`, this sharpens or softens distribution which has huge impact on perfrmonace when used with LAP. Temperature value can be picked on validation set.
* The "277 classes per plate" trick. It might be different from what other participant were doing, but basically we did the following:
1. Search for best **temperature** that maximizes metric.
2. Use **LAP** within experiment to get model predictions.
3. Now we need to decide: within each experiment, what sirna groups should be assigned to each plate. The next step if to just **zero out probabilities** of classes which **does not belong to this group**.
4. Run **LAP** again.

# TTA
* **None**, just average of 2 sites

# Tools and hardware
* **PyTorch**
* 1-2 1080ti most of the time, and about 8 GPUs in last 2-3 days.
* [LAP solver](https://github.com/gatagat/lap)
