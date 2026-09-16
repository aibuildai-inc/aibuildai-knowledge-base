# 21st Place Solution: You don't need cell tiles

Competition: hpa-single-cell-image-classification
Rank: #21
Source: https://www.kaggle.com/c/hpa-single-cell-image-classification/discussion/238474

Hey  everyone, thanks for that awesome challenge! Congrats to everyone :)
I'm verry happy to get my first silver medal on Kaggle and want to share my approach with you.
And thanks to phalanx and his [post ](https://www.kaggle.com/c/hpa-single-cell-image-classification/discussion/217395)on Puzzle-CAM for good inspiration

**General Approach**
Like many of you, I read a lot a weakly-labeled instance segmentation and eventually wanted to go for a image-level training and inferencing method. For this to achieve, a model producing good Class-Activation-Maps was needed so I decided to try [Puzzle-CAM](https://arxiv.org/abs/2101.11253) and do some mapping magic for inferencing to get probabilities from my CAMs. 

**Training**
I trained according to the Puzzle-CAM paper with each images being tiled to four single images and considering the full-image CAMs versus the tiled-image CAMs in a loss function. I used a ResNest-101 and an EfficientNet-B4 with the according GAP Layers added and Focal Loss function. 

**Inferencing**
Here's the interesting part. I'm simply multiplying the CAM of each class with the cell mask of each cell and the class probability the model produces *(using a Swish-Activation to obtain the CAMs gives slightly better results than raw CAMs or ReLU)*. This gives very large class activated values for each class for each cell, which have to be mapped to real class probabilities and I used two approaches for this:
1. standardize the values of each image using `sklearn.preprocessing.StandardScaler` and applying a sigmoid function to these values (works surprisingly good)
2. Do the inferencing on the single-class labeled train data to get the raw values and train a gradient boosting regressor to learn the according label (0..1) for each class (to make sure, that the right mapping function, that might be different from the sigmoid function, is found)

In the end I combined both approaches. 

Now enjoy some nice graphics showing my approaches (click links for higher res) :)


[TRAIN](https://images2.imgbox.com/e7/85/HVh20eFe_o.jpg)


[INFERENCE](https://images2.imgbox.com/fd/b1/4bEYABtz_o.jpg)
