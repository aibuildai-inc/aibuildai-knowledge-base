# 15th place solution [0.45 private LB]

Competition: google-ai-open-images-object-detection-track
Rank: #15
Source: https://www.kaggle.com/c/google-ai-open-images-object-detection-track/discussion/64633

Software
========

Windows 10 + Python 3.5 + Keras 2.2 + Keras-RetinaNet [0.4.1]: https://github.com/fizyr/keras-retinanet

Main approach
=============

All classes were split on 5 levels depends on children level. As the basis I used official classes hierarchy: https://storage.googleapis.com/openimages/challenge_2018/bbox_labels_500_hierarchy_visualizer/circle.html

 - **Level 1**: 443 classes - no children, have maximum impact on score
 - **Level 2**: 46 classes - have children only with level 1
 - **Level 3**: 4 classes: 'Seafood', 'Watercraft', 'Insect', 'Carnivore'
 - **Level 4**: 4 classes: 'Vegetable', 'Land vehicle', 'Reptile', 'Invertebrate'
 - **Level 5**: 3 classes: 'Furniture', 'Vehicle', 'Animal'

**Note 1**: It’s possible to split classes on 3rd, 4th and 5th levels differently. Only “Animal” class has real 5th level.

**Note 2**: 3rd, 4th and 5th don’t have large impact on final score so I didn’t really tune these models.

**Note 3**: Models of 2-5 levels optional, since we can generate predictions for them using Level 1 model. We just need to duplicate boxes for their children and use some NMS algo on them since there will appear some duplicates. Score for this approach will be slightly lower than using separate 2-5 level models (as I remember correctly ~0.01-0.02 lower).

Why we need the split? Why we don’t train using all 500 classes?
================================================================

In process of dataset analysis I found out that there are some images which marked up only on higher level classes. The most explicit representatives are the Person class (2nd level) and Man, Woman, Boy, Girl (1st level). What's the problem? Images marked with the Person class do not have markup for Man, Woman, Boy and Girl. If we use these images for training at once for all classes - the model will be confused in the classes Man, Woman, Boy and Girl. You can throw out images for Person, but then it makes no sense to train this class as part of the overall model, but it's easier to generate markup for Person in the inference step (using boxes for Man, Woman, Boy and Girl).

To train 2-5 levels models, we can use more data, including images marked for example by Person besides images including subclasses (Man, Woman, Boy and Girl) and this improves the result.

Preparing data for training
============================

**Level 1**: All images containing markup for one of the 443 classes were added + images were added that contained the markup of the first-level classes not included in Challenge 500, as negative samples. Excluded all images containing markup of the higher levels in order to reduce the number of potential False negative (that is, the markup should be, but it is not in the training set).

**Levels 2 - 5**: Images for each class contain both markups directly for this class, and for all children classes. Excluded images containing the higher level classes. Added images containing disjoint classes as negative samples.

Validation is rather slow on RetinaNet. I did a separate validation for the training process, where for each class there were about 25 images containing this class. Thus, I balanced the classes a bit and reduced the time of the validation. My validation during training was about 6.5K images instead of 41K.

Best models
===========

    1) Keras RetinaNet + ResNet152 Image size ranges 600-800 px
    mAP on small validation: 0.5028
    mAP on full validation: 0.384009
    LB score*: 0.47441
    
    2) Keras RetinaNet + ResNet101 Image size ranges 768-1024 px
    mAP on small validation: 0.4896
    mAP on full validation: 0.377631
    LB score*: 0.47549

* - I have not tried individual submissions by levels, just sent a merged result for the best models for all levels.

![enter image description here][1]

![enter image description here][2]

![enter image description here][3]

![enter image description here][4]

Models continue improving when I stopped training them, so I think there is some room to increase score.

Process of training and inference
=================================

Keras-RetinaNet already has a implemented generator for training on Open Images Dataset (OID). But in the current form it is not very suitable. I made several changes to it:

 - I added support for “empty” class - for images with negative samples that do not contain any boxes.
 - In the current generator there are no augmentations associated with the color, I added a random change in the intensity of the channels - this gave a good increase in validation score. And it feels like augmentation set in Keras-Retinanet is not enough for training a strong model.
 - Most important, selecting just random images for the batch will work poor at random. I replaced it with the following method:
- before the start of training for each class (including "empty") we create a list of images that contains boxes for the class.
- during the training, to add the next image to the batch, we first randomly select a class, then randomly select the image from the image list for the class. Thus, we achieve more or less uniform training by classes. Strictly speaking, the distribution is still not quite uniform because the images usually contain boxes for several classes at once.
 - From small things: I increased values ​​for augmentations transform-generator, especially scale. I changed the value of factor from 0.1 to 0.9 in ReduceLROnPlateau because the learning rate dropped too fast.
 - At the stage of the convert model for inference, in the FilterDetections layer in RetinaNet - reduced the score_threshold from 0.05 to 0.01 and tried to increase the number of boxes from 300 to 500. This gives more flexibility in the ensemble stage. Also note that the default model RetinaNet for Inference already contains NMS in the FilterDetections layer with nms_threshold = 0.5 and there is a feeling that you can play with this parameter.
 - I restarted training several times with a large LR = 1e-5, when LR fell too low. And each time the model became better. But the experiment is not very clean, because every time I changed the augmentation parameters.

Ensembles
=========

 - For each model, I made a prediction on the image and its horizontal mirror.
 - RetinaNet has internal layer with made NMS on single image prediction (it can be switched off to get the full set of boxes, but I didn’t try it).
 - I merged boxes using two methods:
- standard NMS (https://github.com/rbgirshick/fast-rcnn/blob/master/lib/utils/nms.py), the optimal threshold I found was about 0.75. 
- second method is my own heuristic. It works better on LB. Heuristics included a weighted addition of boxes, which changed their coordinates and Confidence score.
 - I also tried Soft NMS [https://github.com/bharatsingh430/soft-nms] last 3 days of competition, but it works almost the same as default NMS and worse than my ensemble approach. Probably I missed something.

My ensemble approach in short
=============================

 1. On input we get set of boxes from different N models
 2. Set “Init_weight” = 1/N and initialize “result” set of boxes as empty list
 3. Add all boxes for best model with “init_weight” weight to “result” list. For all other boxes of other models try to find best matching box in result using IOU. If box with IOU &gt; THR (0.55) exists in “result” then merge best found box in result with it using weighted average for coordinates and confidence score. Increase weight of this box by Init_weight value. Otherwise add this new box to “result” with “init_weight” weight.

Other solutions
===============

At the first stage I tried to solve the problem simply on pretrain models without any retraining:

 - RetinanNet Pretrain Coco: https://github.com/fizyr/keras-retinanet/releases - gives an approximately 0.13 on LB, if the classes between OID and COCO are correctly matched.
 - On the pretrain from here: https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb using this model: faster_rcnn_inception_resnet_v2_atrous_oid_2018_01_28 you can get ~0.25 on LB.

Observations and small tricks
=============================

 1. Because of the metric nature, it is better to output the maximum number of rectangles even with low confidence score.
 2. Validation works so-so. In most cases, it is worse than LB. Most likely this is due to two factors: the distribution in kaggle test is very different from the distribution for validation. Classes with a small number of elements affect the score as much as others.
 3. The limitation on the size of the CSV file on Kaggle in this task is ~2GB. This fact didn't allow to submit more boxes with low confidence score.

Proposed dataset improvement
============================

In case we will have similar competitions next years:

 - To simplify the task organizers should improve the training dataset, so that there are no situations when there is a markup for level 2 and there is no markup for level 1. 
 - It would be good to have a full markup for each image with all the boxes including the parent classes. To avoid discrepancies / errors, etc.
 - It’s better to use actual class names like "Ambulance" instead of /m/012n7d
 - A little confusing is the presence of the flags like “isGroupOf” - are there any images of such type in the Test set or not? I eventually excluded these images from training and still not sure if it was right thing to do.
 - It seems to me that the current metric is not very good due to the fact that classes with a very small number of boxes influence just like classes with millions of boxes. Small mistakes can lead to significant score changes.

Code and PreTrained models
==========================

GitHub repository: https://github.com/ZFTurbo/Keras-RetinaNet-for-Open-Images-Challenge-2018


  [1]: https://storage.googleapis.com/kaggle-forum-message-attachments/379163/10220/ResNet101-loss.png
  [2]: https://storage.googleapis.com/kaggle-forum-message-attachments/379163/10218/ResNet101-valid-mAP.png
  [3]: https://storage.googleapis.com/kaggle-forum-message-attachments/379163/10219/ResNet152-loss.png
  [4]: https://storage.googleapis.com/kaggle-forum-message-attachments/379163/10217/ResNet152-valid-mAP.png
