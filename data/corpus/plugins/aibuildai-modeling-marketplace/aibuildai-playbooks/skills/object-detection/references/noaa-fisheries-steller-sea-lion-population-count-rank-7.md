# AdaBoost, SSD: object detection for lions counting

Competition: noaa-fisheries-steller-sea-lion-population-count
Rank: #7
Source: https://www.kaggle.com/c/noaa-fisheries-steller-sea-lion-population-count/discussion/35462

We made a bet with my friend, how far we can go in this challenge (and two month of life have blown away). The problem looks like a type of crowd counting task. Recent solutions prepare density maps for dotted crowd annotations, train network to predict them, and then sum the density in each pixel to obtain the final number. However, I've decided to use object detection approaches, and my friend started with segmentation...
___
My path:

1) Classic ml - AdaBoost

2) SSD on full images (since it is faster than faster rcnn)

3) SSD on tiles

For object detection boxes is needed, so decided to use squares around the dots coordinates (size was selected manually, OpenCV `inRange` did the work with dots coordinates extraction).

As for now DL is a mainstream with state of art results, but around 3 years ago, *the best off-the-shelf classifier* **AdaBoost** crushes the leaderboards, so cannot skip to evaluate it. One of the fastest and successful AdaBoost pipelines is boosted decision forest with aggregated channel features (thx to Piotr Dollar for this milestone in object detection). Thus trained 3 models: adult, subadult males and pups. This gives ~20 on public LB *(it took around of an hour to evaluate one model on all test set, on CPU)*. The recall is almost 100%, however, due to lions rotations, models learned to find circle-shaped objects, so there were a lot of false positives on stones.

DL's time. Single Shot Multibox Detector was selected for the speed reasons and good accuracy. Scaled the images to fit the GPU memory (~1200x800) and run with VGG backbone out of the box. DL features give ~18 on LB, mostly due to better (than mean values) work on females and juveniles, testing time is also ~1 hour, but on GPU. Surprisingly, AdaBoost with ACF did well on adult, subadult males, the difference was ~0.1-0.3 compared to SSD. The issues:

* pups detected badly

* can't distinguish between females and juveniles

What tried to beat this (without any success):

* Better features - construct a hyper feature from conv3_3 &amp; conv4_3 &amp; conv5_3 to integrate the details from shallow layer and context from deeper.

* Learn the image scale - average pool of feature maps + fc128 + fc16 and concatenate these 16 learned channels with feature maps for final prediction. So, I also did nothing with the scale problem.

* Tried multitask loss - added regression output to SSD classification and localization. Regression produced more or less reasonable values for adult, subadult males, but for classes with huge deviation in number of lions per image obtained `nan`.  Looks like need to spend more time here =), and use small tiles instead of full images.

So, gave up with sophisticated methods, tile the train data. Used huge tiles 1000x1000 for speed reasons and to avoid border effects. It helped to detect pups better, gives ~17 on LB (during training and testing images were resized to 3000 by the shortest side, then tiles extracted, did flip augmentation), runs ~8 hours on GPU:
![Picture 1][1]
![Picture 2][2]
The rest 1.5 point is stacking models (spent ~100 submissions while doing this).

Thanks for everyone who did involve in this competition. It was fun!

  [1]: http://i67.tinypic.com/juhicw.png
  [2]: http://i63.tinypic.com/nv76ns.png
