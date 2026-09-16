# 17th place solution

Competition: google-ai-open-images-object-detection-track
Rank: #17
Source: https://www.kaggle.com/c/google-ai-open-images-object-detection-track/discussion/64747

Congrats winners and all participants!

My solution is very simple.

**First-Stage** Train ResNet50-FPN Faster RCNN model with all training data (480000 iters, 8 batches). 
I trained this model to predict all 500 classes without descrimination of label's hierarchy.
ResNet50-FPN Faster RCNN code is (https://github.com/Hakuyume/chainer-fpn).

**Second-Stage** Train this model with sampling training data. (230000 iters, 8 batches)
I down-sampled only the classes contained in many images(&gt;= 5,000 images)
First-Stage, it took a long long time with my machine, so I tried training with this down-sampled data to train rare classes well.

**Third-Stage** Predict test images (normal and horizontal flip).
I added expand labels to predict result based on label's hierarchy.
e.g.

label, score, xmin, xmax, ymin, ymax

FootBall Helmet, 0.5, 0.1, 0.3, 0,2, 0.4

-&gt;

FootBall Helmet, 0.5, 0.1, 0.3, 0,2, 0.4

Helmet, 0.5, 0.1, 0.3, 0.2, 0.4

https://storage.googleapis.com/openimages/web/object_detection_metric.html

**Fourth-Stage** combine expanded predicts (normal and horizontal filp) with non-maximum supressions.
