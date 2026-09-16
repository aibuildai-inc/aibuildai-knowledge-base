# #3 BR POWER - Solution

Competition: state-farm-distracted-driver-detection
Rank: #3
Source: https://www.kaggle.com/c/state-farm-distracted-driver-detection/discussion/22631

First of all congratulations to Jacobkie and Z_B_C for winning such amazing competition. @Z_B_C we almost draw.
Also I would like to thanks Kaggle and StateFarm for such a unique competition. We leaned a lot and it is my first DeepNets competition win ;-D

Our solution is most based in the "Time" feature. Just watch the movies in the link and try to figure out why: https://www.kaggle.com/titericz/state-farm-distracted-driver-detection/just-relax-and-watch-some-cool-movies

All pictures in trainset are taken in sequence so we explored that characteristic. Trainset is given in the correct sequence, but Testset not. So taking into account that subsequent images are very close to each one (driver position, light, shadows, vehicle external objects, etc...), we decided to try nearest neighbors on all images. And for our surprise it presented very good results catching the nearest images in the correct trainset sequence. The first neighbor have a subject hit ratio of 100% and class hit rate of 99,5% in trainset. So we used a blend of the 20 nearest neighbors in our solution.

We trained 8 CNN models, but for our final submission we used only 4. All models trained over 5 folds CV:

1)- resnet-152 caffe, CV: 0.31 LB: 0.27. Original dataset, no augmentation

2)- resnet-152 caffe, CV: 0.36 LB: 0.31. Original dataset, some augmentation (under-tuned?)

3)- resnet-152 torch, CV: 0.223 LB: 0.181. Modified images 1, no augmentation

4)- VGG-16 Keras, CV: 0.30 LB: 0.28, . Modified images 2, 2x augmentation

=>Modified Images 1: Took current image and 5 nearest neighbor images. For each one of the 3 RGB channels I replaced the image by:

Channel R:  (current image + nearest1)/2

Channel G:  (nearest2 + nearest3)/2

Channel B:  (nearest4 + nearest5)/2

It built very redundant images. Also on these images we tried to center the steering wheel via regression.


=>Modified Images 2: Took current image and 5 nearest neighbor images. For each one of the 3 RGB channels I replaced the image by:

Channel R:  current image - nearest1

Channel G: nearest2 - nearest3

Channel B:  nearest4 - nearest5

It build very redundant images and tries to catch drivers movements. 

So these 4 models presented high diversity between then and predictions are very stable at Level 1 training.


So we used Level 1 prediction of all 4 models to ensemble at Level 2. But this time merging all 20 neighbors predictions of each image. So each image have its own prediction + 20 neighbors predictions. For Level 2 training we used scipy minimize function and created a custom geometric average function to minimize logloss of all models. That architecture improved CV of each model:

1)- resnet-152 caffe, CV: 0.31 =>  0.192

2)- resnet-152 caffe, CV: 0.36 => 0.276

3)- resnet-152 torch, CV: 0.22 => 0.180

4)- VGG-16 Keras, CV: 0.30 => 0.192


Our final solution is and weighted geometric average of these 4 models and CV score is about 0.116 and Class hit Ratio of 96.4%

Also we found via cross-validation that replacing all prediction < 0.00001 to zero improved our scores in CV, but we didn't used that in our last submission. If we had choosen it we would finished #1  :-/


That's it and...

Congratulation again Jacobkie for its late huge jump to #1. And congrats to all top10 teams that didn't overfitted and to everyone that spent much time in this one. Lots of learnings to all.

Thanks again!
Giba

obs. don't forget to upvote the post and script if you like it ;-P
