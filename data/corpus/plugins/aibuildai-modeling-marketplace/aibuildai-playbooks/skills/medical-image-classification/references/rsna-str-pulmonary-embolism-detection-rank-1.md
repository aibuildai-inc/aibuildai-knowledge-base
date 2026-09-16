# 1st place solution with code

Competition: rsna-str-pulmonary-embolism-detection
Rank: #1
Source: https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection/discussion/194145

Congratulations to all the winners! Thanks to Kaggle and RSNA for hosting this competition and presenting us this interesting problem. The data size is big and of high quality and there is no shakeup. I’m glad I can win this one and I have learnt a lot during this journey.
Special thanks to @vaillant for providing the topic introduction and useful input processing code. Also, credits should go to last year’s RSNA winners, lots of their ideas are incorporated in my solution.
https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/discussion/117242

My solution is described below. 
Code: https://github.com/GuanshuoXu/RSNA-STR-Pulmonary-Embolism-Detection
Inference kernel: https://www.kaggle.com/wowfattie/notebook6fff7ff27a?scriptVersionId=45476524

# Preprocessing

Early after I joined this competition, I noticed that increasing input image size from 512x512 to 640x640 improves the modeling performance. By browsing the training images, I further noticed that the lungs did not occupy large and consistent portions of the images. This is inefficient because we know input size matters and it’s not worthy to waste computing time on irrelevant things in the images, and this could also give the modeling unnecessary difficulty to learn large scale and shift invariance. So, it’s necessary to have a high-quality lung localizer. There are some existing pretrained lung localizer online, I did not try them because according to my observation it’s easy for a CNN to accurately localize the lung area from images as long as we have the bbox labels of the lungs. So, I annotated the train data and built a lung localizer with the bboxes and Efficientnet-b0 as the backbone. For simplicity I only annotated four images per study. The training and prediction process were also on only four images per study to save time. Some examples of this preprocessing are given below. The localizer is very robust even in some relatively difficult conditions. The idea of preprocessing the input is partly inspired from last year’s 2nd place solution.



# Training/validation split

Since the provided data are big and of high quality, we don't have to do cross validation, a single training/validation split is reliable enough. In this competition, I randomly set aside 1000 studies for validation and used the rest 6200+ studies for training and hyperparameter tuning. For final LB submission I re-trained my models with the full training set and the optimized hyperparameters.

# Image-level modeling

I used the same 2-stage training strategy as in last year’s RSNA competitions. For image-level modeling, the 3-channel input was the PE windows of the current image and its two direct neighbors. Using neighboring images has proved to be effective in last years 1st and 3rd place solutions. My experiments also confirmed that this input setting outperformed single images with 3 types of windows.

Apart from predicting image-level labels, this year we are given various study-level labels. At first glance it appeared to me that, because the input of the study-level models are image embeddings, we need to use these study-level labels during image-level modeling so that the following study-level model could have sufficient knowledge to model and predict them. But after I tried lots of combinations of them and various loss masking tricks, the best performing  model in both the image-level and study-level stages was still the one trained with the image-level labels only. I’m a little puzzled how the image embeddings are encoded with the study-level labels, for example, the exact position labels (center, left, right) and the more refined acute and chronic, when the image-level models were not trained using any of those labels.

The training loss was the vanilla BCE loss with linear lr scheduler. No special data sampling was applied. I found that a single epoch through the train data was the optimal for my settings. The best augmentations were 

```
albumentations.RandomContrast(limit=0.2, p=1.0),
albumentations.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20, border_mode=cv2.BORDER_CONSTANT, p=1.0),
albumentations.Cutout(num_holes=2, max_h_size=int(0.4*image_size), max_w_size=int(0.4*image_size), fill_value=0, always_apply=True, p=1.0),
```

My final ensemble were with one serexnext50 and one seresnext101. Their respective validation performance for image-level PE prediction was

```
                     Loss      AUC
seresnext101        0.079    0.964
seresnext50         0.080    0.962
```

Other good backbones are inception_resnet_v2 and efficientnets. Densenets and resnexts performed a lot worse. Input were resized to 576x576 after the lung localization, this was the largest size the models could finish running in the 9 hours.

# Study-level modeling

Image embeddings of dimension 2048 served as the input to a RNN for both image-level and study-level modeling.  

One thing we need to handle was that the number of images each study has could vary from 100+ to 1000+. As we don’t know the information of the private test data, it was hard to predefine an input sequence length for our RNN model if we want to predict all the images. Stacking all the images into a 3-D array and resizing it along the z-axis before generating image embeddings is an option, but it was not compatible to my inference pipeline. For convenience, I swapped the order of embedding generation and resizing, in other words, I chose to resize the features instead of images. For example, given a study which has N images, the input feature shape is Nx2048. If the max sequence length limit in the RNN is M, the cv2.resize function is applied to resize features to Mx2048 if N>M, otherwise if N<M zero-padding is used. The image-level labels and the predictions are zoomed in and out in the same way during training and inference. To find the best M, I ran a search in the step size of 32, and M=128 gave the best performance. In the train set, the majority of Ns is in the range of 200-250. This means downsizing across the z-pos first before sequence modeling improves the performance. In my final models, I actually set m=192 because I believed there might be more big Ns in the private test data.

Inspired from last year's 2nd place solution, I also computed the difference of embeddings between current and the two direct neighbors and concatenate with the current features. So the input size was expanded to 2048x3. 

The exact RNN architecture is not very important, I settled down to only a single bidirectional GRU layer, with the study-level labels predicted by a concatenated attention weighted average pooling and max pooling over the sequence. My local validation loss was around 0.18, I have no idea why it is much higher than the LB scores.



# Postprocessing

The main purpose of the postprocessing step is to satisfy the consistency requirement of the labels. Since this consistency requirement agrees with how the data was labeled, a careful postprocessing could improve the performance. In my case, the local validation has a tiny improvement after postprocessing. The brief workflow of the postprocessing is 

```
for each study:
    if the original predictions satisfy the consistency requirement
        do nothing
    else
        change the original predictions into consistent positive predictions, and compute loss between them
        change the original predictions into consistent negative predictions, and compute loss between them
        choose from the positive and negative predictions based on which causes the smaller loss
```


The weights of the loss function is almost same as the competition metric, except that the  q_i of image loss weight is replaced by a fixed 0.005 because we don’t have the ground truth of the test data. Code of this postprocessing can be found in my inference kernel.
