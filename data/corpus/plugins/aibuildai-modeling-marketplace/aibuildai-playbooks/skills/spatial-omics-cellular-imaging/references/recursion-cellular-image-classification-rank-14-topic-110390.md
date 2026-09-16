# 14th place solution, [0.989] Colab and Kaggle kernels

Competition: recursion-cellular-image-classification
Rank: #14
Source: https://www.kaggle.com/c/recursion-cellular-image-classification/discussion/110390

I want to thank Recursion Pharmaceuticals and Kaggle for organizing this competition. I also want to thank everyone who participated and especially [nosound](https://www.kaggle.com/zaharch) for explaining and providing sample code for leak exploitation.

My solution was developed using only freely available computational resources. I used Colab at the start of the competition, then briefly switched to Kaggle kernels up to the point when Kaggle enforced GPU time limitations and then Colab again. My final submission was a single densenet201-based model.

**Tools**
- PyTorch with Apex and a bit of Ignite.

**Data**
- 6x512x512. First I've tried to use smaller resolutions, but understood fairly quickly that full-size images are necessary for maximum performance.
- For augmentation I've used rotations, horizontal flip, cutout and random brightness/contrast. I've also used 4 images rotated by 90 degrees for TTA.

**Model**
- densenet201 with ArcFace loss.
- separate 1108 classes for every cell type.

**Training process**
- AdamW and cyclical learning rate with restarts, triangular schedule.
- validating on 10% of train dataset to find a point where model begins to overfit. Training on the whole dataset up to the cycle that showed improvement on validation.
- using the last checkpoint in a cycle for prediction.
- averaging distances for two sites.

**Post-processing**
- using leak. Getting maximum prediction combination using lapjv for every plate.

Pseudo-labeling and blending different models' predictions would have likely boosted my score, but unfortunately there was not enough time and computational resources.

**Congratulations to the winners!**
