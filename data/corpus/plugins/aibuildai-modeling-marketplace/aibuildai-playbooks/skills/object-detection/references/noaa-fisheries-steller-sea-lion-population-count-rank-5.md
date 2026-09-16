# R&D&G team solution, 5th place

Competition: noaa-fisheries-steller-sea-lion-population-count
Rank: #5
Source: https://www.kaggle.com/c/noaa-fisheries-steller-sea-lion-population-count/discussion/35491

Congrats to outrunner, Konstantin and bestfitting!

A short overview of our team solution:

We estimated image size, implemented u-net line network and with sum of heatmaps as resulting count.

Our image scale estimation solution:

Firstly I have labelled the scale of the training dataset images.
To do this, I have done a simple application to display all sea lions from the image next to lions from the reference image, and adjusted scale of lions until it visually matches for all classes:
https://ibb.co/mWg5oQ

Instead I trained the SSD network to detect sea lion rects using the average sea lion size for known  image scale and used predicted sea lion rects as input to xgboost network, predicting actual image scale.

So training pipeline looks like:

1. Label image scale for train dataset

2. Extra scale augmentation, prepare set of sea lion squares

3. Train SSD, predict sea lion squares

4. Combine results of SSD, like mean of predicted crops mean size, size of different percentiles of rects sorted by detection confidence, the same per class, etc and trained XGBoost to predict image scale

Prediction pipeline:

1. Downscale image 2x as sea lions at some images are over 3x larger than on the train dataset

2. Predict sea lion rects with SSD and image scale with XGBoost

3. Scale test to scale predicted with step 2, it should be the better match than original 0.5x

4. Predict scale with SSD and XGBoost again

Applying estimated scale allowed to improve score from ~23 to 15.9.

Distribution of image scale of the train and test datasets (on this plot - how much image has to be rescaled to match train/0.jpg):

https://ibb.co/i3X38Q

For some reason the test dataset is significantly different from the train dataset, image scale is approx 2x larger.



To count sea lions I combined the imagenet pretrained networks like Resnet and VGG with decoder part of U-net.

I used pretrained and fine tuned on sea lion crops Resnet50 network,
attached 3 stages of U-net like decoder to network output and internal layers.

The resulting resolution was 4x lower than resolution of original image.
I tried to use original U-Net or added more stages to increase resolution, but results were worse. Adding of residual blocks to U-Net decoder allowed to improve the results.

I marked each lion as 3x3 square regardless of class and predicted heatmap with softmax loss for 5 sea lion classes + 1 background class. 

My idea was - if logloss/softmax has minimum when output is equal to probability of particular  class, the count of sea lions should be equal to sum of per class heatmap / area of marker used for training. It worked reasonably well, on images with wrong scale the count shifted between classes as network learned lion sizes but the total count matched.

examples of heatmaps, number is predicted count, as sum(heatmap)/(3*3)

https://ibb.co/ndUmv5

https://ibb.co/jg6ANk

For the final submission, Gilberto has trained a few 2nd level models we averaged together with predictions of Resnet50 and VGG16 based models.
