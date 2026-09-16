# 3rd place solution (U-Net + Dilated Conv)

Competition: carvana-image-masking-challenge
Rank: #3
Source: https://www.kaggle.com/c/carvana-image-masking-challenge/discussion/40199

Thanks for hosting such an exciting competition! I was really enthusiastic for spending my time for this competition! And thanks for helpful code by @Peter and excellent ideas by @HengCher Keng. I learned a lot from them.

The competition repository is here. I put two scripts (My network script &amp; loss functions script) in it.
https://github.com/lyakaap/Kaggle-Carvana-3rd-place-solution

## My solution overview

* I used 1536x1024 &amp; 1920x1280 resolution.

* I used modified U-Net. It has several dilated convolution layers in bottleneck block. (i.e. where the resolution of feature maps are lowest)

Detailed figure of my network architecture is here.

![my_network][1]

The best score of this model is **0.997193** only around 8.5 million parameters. (trained one of 6 folds, no TTA &amp; no ensemble, input resolution: 1920x1280)
Averaging two predictions(TTA, original image &amp; flipped image) by 0.997193 model reached **0.997222**. They are ranked 6th place and 5th place on LB respectively!

I tried normal convolution layers instead of dilated convolution layers in bottleneck block, and its score is significantly lower than using dilated convolution. (normal: 0.9905, using dilated conv: 0.9918 @256x256)

I also tried parallelized dilated convolution layers instead of stacking them, but it gave me lower score than stacked architecture.

* Optimizer: RMSprop lr = 0.0002, reducing learning rate by using ReduceLROnPlateau() that is Keras callback function. Reducing factor is 0.2 &amp; 0.5

* Data Augmentation: only horizontal flip. Scaling, Shifting, and Shifting HSV were results of overfitting for me.

* Batchsize: 1, and no BN.

* Training whole time on single model takes around 2 days.

* Pseudo Labeling: learning simultaneously or only using pretraining phase.

* Loss function: bce + dice loss (I also tried weighing boundary pixel loss, it gave similar result. Fear of overfitting, I finally decided not to use it.)

* Ensemble: 5 fold ensemble @1536x1024 + 6 fold ensemble @1920x1280, weighted average. I weighted by LB ranking in my submissions.

* TTA: only horizontal flip.

* Adjusting threshold: I decided threshold which gives best score on validation set. I set the threshold to 0.508. In LB, it makes score improving only 0.000001.

## Other

One of the best contributer of improving score is training on pseudo labeling data. I think why pseudo labeling contribute so much is the amount of test data, and we can get predictions close ground truth.

I tried post processing by using pydensecrf for only difficult to mask car images. But it gave me no improvement.
As for how to choose "difficult images", I calculated multi class version of dice coeficient (I'm afraid that I shouldn't say so) of predictions by several models.


  [1]: https://kaggle2.blob.core.windows.net/forum-message-attachments/225523/7428/network.png
