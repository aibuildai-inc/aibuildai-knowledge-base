# 9th place solution

Competition: intel-mobileodt-cervical-cancer-screening
Rank: #9
Source: https://www.kaggle.com/c/intel-mobileodt-cervical-cancer-screening/discussion/35104

Used tools
--------------

Windows 10 + Python 3.4 + Keras 1.2 + Theano 0.9

ZF_UNET_224 (version 1)
-----------------------

To find important parts of images I used [my own version of UNET][1]. First UNET was trained on segmentation of train images, [which I made by hands][2] with Sloth. In this segmentation I tried to cut all elements which are totally useless like speculum and everything outside it. The predictions look nice:

![enter image description here][3]

ZF_UNET_224 (version 2)
-----------------------

Second UNET was trained on rectangles [provided by Paul][4]. This net finds the main region of interest. The predictions look even better.

![enter image description here][5]
 
I didn’t use these predictions directly. I only extract bounding box rectangles for each image to form rectangles.csv file which I used in later training process.

Zoo
---

I used the following set of CNNs: VGG16, VGG19, RESNET50, INCEPTION_V3, SQUEEZE_NET, DENSENET_161, DENSENET_121. The training process for them was mostly similar. 5 KFold validation, vary only learning rate parameters and batch size. For each model I obtain: train OOF predictions with same length as number of training images and test predictions.

Augmentations
-------------

I think augmentation was important key for this problem, since we had very small amount of data. I used: 

 1. Random crops based on rectangle.csv generated from UNETs. These crops were in very big range from UNET_v1 prediction to UNET_v2 predictions. 
 2. Random perspective transformation 
 3. Random rotations (mean color or border reflect at random) and mirroring 
 4. Lightning change 
 5. Rare random blur

Basically neural nets never see the same pictures during training.

Batch generator
---------------

For learning process I used fit_generator function from Keras, which I recommend to use for everyone. You only need to create your own batch_generator function. No need to store many different images in memory or on HDD.
In my batch generator I add some fraction of images from “additional” folder ~25%. It greatly improves validation and leaderboard score. And we all know now it’s because of leakage of test images in additional folder.

Ensemble
--------

For ensemble I used XGBoost blender. Final solution was average on 500 XGBoost iterations with different seed and random run parameters. 

Submission
----------

The only difference for my 2 final submissions was to train with usage of “additional” images or without them. My final models used “train” and “test_stg1” images for training.

Submission v1 (with additional): Validation score: 0.57457 Private LB: 0.88856

Submission v2 (without additional): Validation score: 0.64254 Private LB: 0.83209

I believe low scores on private LB depends on bad data preparation, mostly because of mislabeling. We could clean it by hands, but in case private test set had the same labeling quality it would make everything worse.

Code
----

You can find my code as it was prepared for Kaggle [on GitHub][6]


  [1]: https://github.com/ZFTurbo/ZF_UNET_224_Pretrained_Model
  [2]: https://kaggle2.blob.core.windows.net/forum-message-attachments/194936/6708/polygons.json
  [3]: https://kaggle2.blob.core.windows.net/forum-message-attachments/194936/6709/ZF_UNET_v1_predictions.jpg
  [4]: https://www.kaggle.com/c/intel-mobileodt-cervical-cancer-screening/discussion/31565
  [5]: https://kaggle2.blob.core.windows.net/forum-message-attachments/194936/6710/ZF_UNET_v2_predictions.jpg
  [6]: https://github.com/ZFTurbo/KAGGLE_CERVICAL_CANCER_2017
