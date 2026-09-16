# 20th place solution with code

Competition: landmark-recognition-2019
Rank: #20
Source: https://www.kaggle.com/c/landmark-recognition-2019/discussion/94645#latest-549974

Hola,

TL;DR: SE-ResNext50, softmax 92k classes, 224x224 crops and data cleaning on the test set using pretrained Mask R-CNN and ImageNet classifier. The code is here: https://github.com/artyompal/google_landmark_2019

So, my solution is simple, it's a single model (SE-ResNext50), single fold. My code is very similar to my kernel: https://www.kaggle.com/artyomp/resnet50-baseline. I used 224x224 crops from 256x256 images. I trained only on classes with &gt;= 10 samples (92740 classes). Surprisingly, it wasn't much slower than training on 18k or 35k classes (maybe 20% slower).

This is the config of my model: https://github.com/artyompal/google_landmark_2019/blob/master/models/v1.3.8.seresnext50_92740_classes.yml So, it adds a bottleneck layer with 1024 neurons, uses SGDR from https://arxiv.org/pdf/1608.03983.pdf and rectangular crops from https://arxiv.org/pdf/1812.01187.pdf. 

It resulted in 0.128 on public LB after almost 5 days of training on a single 1080Ti.

Obviously, such a low score was caused by both a noisy train set and a noisy test set. I had no time to clean the train set and re-train, so I only cleaned the test set.

The first stage of cleaning was just No Landmark Detector from this kernel: https://www.kaggle.com/rsmits/keras-landmark-or-non-landmark-identification. I just got rid of predictions where any of top3 results were "not landmark".

The second stage. I noticed that there's a lot of portraits, selfies, and pictures of cars in the test set. So I took the pretrained Faster R-CNN from the new TorchVision release (checkout object_detection.py) and got rid of images with too many humans or cars (confidence &gt;0.5, total area &gt; 0.4). Check this out: https://github.com/artyompal/google_landmark_2019/blob/master/patch_sub2.py

The third stage. I noticed that there's a lot of random images: dogs, helicopters, flowers and so on. So I took a pretrained ImageNet classifier. I used ResNet50 (I wanted to use something better, but my GPUs were busy doing something else, so I needed a simple network; ideally, I'd take some smarter model). It's in the image_classifier2.py. So I rejected images with any of those classes and confidence &gt; 0.7 (https://github.com/artyompal/google_landmark_2019/blob/master/patch_sub3.py)

The first stage gave me 0.02 on the public LB, but didn't help at all on the private LB. I wonder if the public/private split was made by class? The private part of the stage-2 test set seems to have cleaner data.

In total, before cleaning I had 0.128 on public LB, 0.144 on private LB. After cleaning I got 0.162 on public LB, 0.148 on private LB.

What didn't work: I implemented KNN on top of SE-ResNext50 features and it worked okay locally (GAP 0.2 on all classes), but it performed like crap on the leaderboard (about 0.00002 or something). I must have a bug somewhere.

PS: thanks to @pudae81 for this repo: https://github.com/pudae/kaggle-humpback. I borrowed some code from there.

Thanks for reading this and I hope this helps in future competitions!
Artyom
