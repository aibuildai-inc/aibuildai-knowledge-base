# 0.51276 Public LB Solution

Competition: dstl-satellite-imagery-feature-detection
Rank: #2
Source: https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection/discussion/29829

To solve this problem I used: Python + Keras 1.0.8 + Theano under Windows 10. As hardware I had NVIDIA GTX 980 8 GB. I used only one GPU which worked almost 24h a day during competition. After join the team I used 2 additional GPUs (TITAN 12GB).
In short my solution can be described like this:

 1. My main CNN is modified UNET with input shape (20, 224, 224). It
    has higher depth, with added batch normalization and dropout layers.
    As loss function is used Jacquard Coefficient. I used grid search
    with different parameters and choose the best model with highest
    validation score. 
 2. I create separate models for each class (so 10
    independent models) and tuned them independently. I actually think
    that I lost some information about class interaction this way, but
    it was much easier to tune models. I partially fix it with
    postprocessing step. 
 3. Each model is actually set of K different
    weights obtained with KFold. For most classes I used 5 KFold. For
    class 7 - 2 KFold. For class 9 – 4 KFold. I split train set by image
    ID (there were only 25 images). I made it once by hands
    independently for each class. Each fold contains the same number of
    images with existed class. So it’s actually stratified split.

Data preprocessing
------------------
Each “image object” (with particular img_id) had set of images made with different wave lengths. In total 20 channels. I resized all images to 3360x3360 pixels (3360 = 15*224 and 3360 is closest to panchromatic image resolution).  And join them along the axis. So at the end I had tensor of following shape: 20x3360x3360. I created the pixel masks from given polygons with same size: 3360x3360 pixels.

![Set of images example][1]

![Mask example][2]

I decided to use UNET with input shape 20x224x224. I choose 224 for 2 reasons:

1. 224 = 2*2*2*2*2*7 – I have 5 MaxPooling layers in UNET and on lowest layer has 7x7 pixel size. Which in my experience the best.
2. The same size I could use with pretrained VGG16 or ResNet.

The next step was made independently for each class:

 1. Split input images on 15*15 parts forming tensors of size
    20x224x224. 
 2.	Because of class imbalances we need to increase
    number of cases where mask exists. For some cases like class 10
    (with vehicles) I have around 99% of empty masks after first step.
    So I added more cases using sliding window around non-zero mask
    points.


Final statistics for classes:

    Number of tests for class 1: 14010. Empty files: 4431 Percent: 31.62%
    Number of tests for class 2: 9835. Empty files: 3997 Percent: 40.64%
    Number of tests for class 3: 20548. Empty files: 5238 Percent: 25.49%
    Number of tests for class 4: 12411. Empty files: 2976 Percent: 23.97%
    Number of tests for class 5: 10161. Empty files: 294 Percent: 2.89%
    Number of tests for class 6: 10716. Empty files: 3720 Percent: 34.71%
    Number of tests for class 7: 10328. Empty files: 5504 Percent: 53.29%
    Number of tests for class 8: 11349. Empty files: 5469 Percent: 48.18%
    Number of tests for class 9: 12035. Empty files: 5574 Percent: 46.31%
    Number of tests for class 10: 16173. Empty files: 5336 Percent: 32.99%

UNET requires the distribution to be close to normal. Ranges for different channels was: P_3, P_P, P_M – 2048, P_A – 16384. At first I divide every channel to its maximum possible value. I’ve seen on forum that some people removed some values from the both ends of histogram but I didn’t do it.

Then I needed to calculate mean and stdev. I calculated it for each channel independently, but at the end used the same mean and stdev for whole tensor: 
mean = 0.219613
stdev = 0.110741

Creating models
---------------
![Modified UNET][3]

    ____________________________________________________________________________________
    Layer (type)             Output Shape   Param #     Connected to
    ====================================================================================
    input_1 (InputLayer)      (20, 224, 224)  0                                 
    ____________________________________________________________________________________
    conv2d_1 (Convolution2D)  (32, 224, 224)  5792     input_1
    ____________________________________________________________________________________
    batchnorm_1 (BatchNormal  (32, 224, 224)  64       conv2d_1             
    ____________________________________________________________________________________
    activation_1 (Activation) (32, 224, 224)  0        batchnorm_1        
    ____________________________________________________________________________________
    conv2d_2 (Convolution2D)  (32, 224, 224)  9248     activation_1                
    ____________________________________________________________________________________
    batchnorm_2 (BatchNormal  (32, 224, 224)  64       conv2d_2             
    ____________________________________________________________________________________
    activation_2 (Activation) (32, 224, 224)  0        batchnorm_2        
    ____________________________________________________________________________________
    maxpool2d_1 (MaxPooling2D)(32, 112, 112)  0        activation_2                
    ____________________________________________________________________________________
    conv2d_3 (Convolution2D)  (64, 112, 112)  18496    maxpool2d_1              
    ____________________________________________________________________________________
    batchnorm_3 (BatchNormal  (64, 112, 112)  128      conv2d_3             
    ____________________________________________________________________________________
    activation_3 (Activation) (64, 112, 112)  0        batchnorm_3        
    ____________________________________________________________________________________
    conv2d_4 (Convolution2D)  (64, 112, 112)  36928    activation_3                
    ____________________________________________________________________________________
    batchnorm_4 (BatchNormal  (64, 112, 112)  128      conv2d_4             
    ____________________________________________________________________________________
    activation_4 (Activation) (64, 112, 112)  0        batchnorm_4        
    ____________________________________________________________________________________
    maxpool2d_2 (MaxPooling2D)(64, 56, 56)    0        activation_4                
    ____________________________________________________________________________________
    conv2d_5 (Convolution2D)  (128, 56, 56)   73856    maxpool2d_2              
    ____________________________________________________________________________________
    batchnorm_5 (BatchNormal  (128, 56, 56)   256      conv2d_5             
    ____________________________________________________________________________________
    activation_5 (Activation) (128, 56, 56)   0        batchnorm_5        
    ____________________________________________________________________________________
    conv2d_6 (Convolution2D)  (128, 56, 56)   147584   activation_5                
    ____________________________________________________________________________________
    batchnorm_6 (BatchNormal  (128, 56, 56)   256      conv2d_6             
    ____________________________________________________________________________________
    activation_6 (Activation) (128, 56, 56)   0        batchnorm_6        
    ____________________________________________________________________________________
    maxpool2d_3 (MaxPooling2D)(128, 28, 28)   0        activation_6                
    ____________________________________________________________________________________
    conv2d_7 (Convolution2D)  (256, 28, 28)   295168   maxpool2d_3              
    ____________________________________________________________________________________
    batchnorm_7 (BatchNormal  (256, 28, 28)   512      conv2d_7             
    ____________________________________________________________________________________
    activation_7 (Activation) (256, 28, 28)   0        batchnorm_7        
    ____________________________________________________________________________________
    conv2d_8 (Convolution2D)  (256, 28, 28)   590080   activation_7                
    ____________________________________________________________________________________
    batchnorm_8 (BatchNormal  (256, 28, 28)   512      conv2d_8             
    ____________________________________________________________________________________
    activation_8 (Activation) (256, 28, 28)   0        batchnorm_8        
    ____________________________________________________________________________________
    maxpool2d_4 (MaxPooling2D)(256, 14, 14)   0        activation_8                
    ____________________________________________________________________________________
    conv2d_9 (Convolution2D)  (512, 14, 14)   1180160  maxpool2d_4              
    ____________________________________________________________________________________
    batchnorm_9 (BatchNormal  (512, 14, 14)   1024     conv2d_9             
    ____________________________________________________________________________________
    activation_9 (Activation) (512, 14, 14)   0        batchnorm_9        
    ____________________________________________________________________________________
    conv2d_10 (Convolution2D) (512, 14, 14)   2359808  activation_9                
    ____________________________________________________________________________________
    batchnorm_10 (BatchNorma  (512, 14, 14)   1024     conv2d_10            
    ____________________________________________________________________________________
    activation_10 (Activation)(512, 14, 14)   0        batchnorm_10       
    ____________________________________________________________________________________
    maxpool2d_5 (MaxPooling2D)(512, 7, 7)     0        activation_10               
    ____________________________________________________________________________________
    conv2d_11 (Convolution2D) (1024, 7, 7)    4719616  maxpool2d_5              
    ____________________________________________________________________________________
    batchnorm_11 (BatchNorma  (1024, 7, 7)    2048     conv2d_11            
    ____________________________________________________________________________________
    activation_11 (Activation)(1024, 7, 7)    0        batchnorm_11       
    ____________________________________________________________________________________
    conv2d_12 (Convolution2D) (1024, 7, 7)    9438208  activation_11               
    ____________________________________________________________________________________
    batchnorm_12 (BatchNorma  (1024, 7, 7)    2048     conv2d_12            
    ____________________________________________________________________________________
    activation_12 (Activation)(1024, 7, 7)    0        batchnorm_12       
    ____________________________________________________________________________________
    upsamp2d_1 (UpSampling2D) (1024, 14, 14)  0        activation_12               
    ____________________________________________________________________________________
    merge_1 (Merge)           (1536, 14, 14)  0        upsamp2d_1              
                                                          activation_10               
    ____________________________________________________________________________________
    conv2d_13 (Convolution2D) (512, 14, 14)   7078400  merge_1                     
    ____________________________________________________________________________________
    batchnorm_13 (BatchNorma  (512, 14, 14)   1024     conv2d_13            
    ____________________________________________________________________________________
    activation_13 (Activation)(512, 14, 14)   0        batchnorm_13       
    ____________________________________________________________________________________
    conv2d_14 (Convolution2D) (512, 14, 14)   2359808  activation_13               
    ____________________________________________________________________________________
    batchnorm_14 (BatchNorma  (512, 14, 14)   1024     conv2d_14            
    ____________________________________________________________________________________
    activation_14 (Activation)(512, 14, 14)   0        batchnorm_14       
    ____________________________________________________________________________________
    upsamp2d_2 (UpSampling2D) (512, 28, 28)   0        activation_14               
    ____________________________________________________________________________________
    merge_2 (Merge)           (768, 28, 28)   0        upsamp2d_2              
                                                                    activation_8                
    ____________________________________________________________________________________
    conv2d_15 (Convolution2D) (256, 28, 28)   1769728  merge_2                     
    ____________________________________________________________________________________
    batchnorm_15 (BatchNorma  (256, 28, 28)   512      conv2d_15            
    ____________________________________________________________________________________
    activation_15 (Activation)(256, 28, 28)   0        batchnorm_15       
    ____________________________________________________________________________________
    conv2d_16 (Convolution2D) (256, 28, 28)   590080   activation_15               
    ____________________________________________________________________________________
    batchnorm_16 (BatchNorma  (256, 28, 28)   512      conv2d_16            
    ____________________________________________________________________________________
    activation_16 (Activation)(256, 28, 28)   0        batchnorm_16       
    ____________________________________________________________________________________
    upsamp2d_3 (UpSampling2D) (256, 56, 56)   0        activation_16               
    ____________________________________________________________________________________
    merge_3 (Merge)           (384, 56, 56)   0        upsamp2d_3              
                                                          activation_6                
    ____________________________________________________________________________________
    conv2d_17 (Convolution2D) (128, 56, 56)   442496   merge_3                     
    ____________________________________________________________________________________
    batchnorm_17 (BatchNorma  (128, 56, 56)   256      conv2d_17            
    ____________________________________________________________________________________
    activation_17 (Activation)(128, 56, 56)   0        batchnorm_17       
    ____________________________________________________________________________________
    conv2d_18 (Convolution2D) (128, 56, 56)   147584   activation_17               
    ____________________________________________________________________________________
    batchnorm_18 (BatchNorma  (128, 56, 56)   256      conv2d_18            
    ____________________________________________________________________________________
    activation_18 (Activation)(128, 56, 56)   0        batchnorm_18       
    ____________________________________________________________________________________
    upsamp2d_4 (UpSampling2D) (128, 112, 112) 0        activation_18               
    ____________________________________________________________________________________
    merge_4 (Merge)           (192, 112, 112) 0        upsamp2d_4              
                                                          activation_4                
    ____________________________________________________________________________________
    conv2d_19 (Convolution2D) (64, 112, 112)  110656   merge_4                     
    ____________________________________________________________________________________
    batchnorm_19 (BatchNorma  (64, 112, 112)  128      conv2d_19            
    ____________________________________________________________________________________
    activation_19 (Activation)(64, 112, 112)  0        batchnorm_19       
    ____________________________________________________________________________________
    conv2d_20 (Convolution2D) (64, 112, 112)  36928    activation_19               
    ____________________________________________________________________________________
    batchnorm_20 (BatchNorma  (64, 112, 112)  128      conv2d_20            
    ____________________________________________________________________________________
    activation_20 (Activation)(64, 112, 112)  0        batchnorm_20       
    ____________________________________________________________________________________
    upsamp2d_5 (UpSampling2D) (64, 224, 224)  0        activation_20               
    ____________________________________________________________________________________
    merge_5 (Merge)           (96, 224, 224)  0        upsamp2d_5              
                                                          activation_2                
    ____________________________________________________________________________________
    conv2d_21 (Convolution2D) (32, 224, 224)  27680    merge_5                     
    ____________________________________________________________________________________
    batchnorm_21 (BatchNorma  (32, 224, 224)  64       conv2d_21            
    ____________________________________________________________________________________
    activation_21 (Activation)(32, 224, 224)  0        batchnorm_21       
    ____________________________________________________________________________________
    conv2d_22 (Convolution2D) (32, 224, 224)  9248     activation_21               
    ____________________________________________________________________________________
    batchnorm_22 (BatchNorma  (32, 224, 224)  64       conv2d_22            
    ____________________________________________________________________________________
    activation_22 (Activation)(32, 224, 224)  0        batchnorm_23       
    ____________________________________________________________________________________
    conv2d_23 (Convolution2D) (1, 224, 224)   33       activation_22               
    ____________________________________________________________________________________
    batchnorm_23 (BatchNorma  (1, 224, 224)   2        conv2d_23            
    ____________________________________________________________________________________
    activation_23 (Activation)(1, 224, 224)   0        batchnorm_23       
    ====================================================================================
    Total params: 31459619


LOSS Function:

    def jacard_coef(y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)

Typical example of training process: Class 1 (Fold 2). Parameters: lr=0.05, optim=SGD, rotation=False, dropout=enabled, UNET version=BatchNorm, patience=8, samples train: 9997, samples valid: 4013

![Typical example of training process][4]

Tuning models and validation
----------------------------
The main problem was that training process is very unstable. Sometimes it can go to local minimum or start predicting as mask full image etc. So most of the time I spend on finding optimal parameters to maximize validation score. It was the most time consuming part. I used grid search with following parameters: 

 - Optimizer (Adam, SGD) 
 - LR SGD: (0.05, 0.01, 0.001) 
 - LR Adam: (0.01, 0.001, 0.0001) 
 - Rotation (enabled, disabled) 
 - Type of model: UNET 224x224, UNET with dropout 0.1, UNET with Batch Normalization 
 - Number of samples per epoch (Fraction from ½ up to 1)

Due to limitation of computational power I checked only some of parameters. 
At the end I use mostly Adam optimizer with BatchNormalization version of UNET with rotation enabled. Just tune learning rate.
I use loss function value to stop training with early stopping with patience 8-15 epochs. After I obtain all 5 Folds models I had the predicted segmentation for each of train images. So I was able to predict score for full image set using the same method as it was made on Kaggle Leaderboard. Obtained score was very representative, so increasing the score locally almost always leads to increasing the score on leaderboard. Also I was able to find optimal threshold value for heatmap with this method.

![LS vs LB Scores][5]

Processing the test data
------------------------
Test set consists of 429 images. Each test image was processed separately. Each image was resized and normalized the same way as training data to 20х3360х3360 tensor. 

In the beginning I create two zero arrays HEATMAP and COUNT with 3360х3360 shape. Then use sliding window approach to predict segmentation. All 5 folds predict on image extracted from current position and acquired probabilities sum up to HEATMAP array. COUNT array at image position added by 1. At the end HEATMAP is divided by COUNT to get real heatmap array. After this heatmap was thresholded at value acquired from validation, typically 0.5. On basis of this 2D-array I created polygons with rasterio and shapely.

![Sliding window][6]

Additional ideas to increase the accuracy provided below. Not all of them was used in final submit because processing of test images was very long process, around 8-10 hours for whole test set.


1.	Decreasing sliding step from 112px to 56px and below always increased the accuracy of predictions, but increase the computational complexity as O(N^2). 
- For example class 4 on validation 0.385714 vs 0.403683.

2.	It looks like UNET predict worse on the edges, so it’s good idea to use only central part for prediction. Example for class 5 below. 
- Default score: 0.507446 
- 200x200 center part from 224x224 square + 100 px sliding window step: 0.512844 
- 160x160 center part from 224x224 square + 80 px sliding window step: 0.514557

3.	I usually used threshold 0.5 without checking optimum on validation, but this can be useful too. 
- Class 6 example: 
- THR 0.1: 0.738943 
- THR 0.5: 0.757858 
- THR 0.9: 0.759850

![Masks for class 5][7]

Ensembles
---------
The most obvious ways to do ensembles is to use heatmaps. But out team consisted of 2 people was merged at the last stage of competitions, so we actually don’t have heatmaps for all classes at this stage. So we made ensembles directly on polygons, using UNION and INTERSECTION from shapely. We have different training process and models, so for some classes we had good boost after merge of this type. 

    Class 6 LB: 0.08149 + LB: 0.08103 (Intersection) Score: 0.08179
    Class 8 LB: 0.03890 + LB: 0.05322 (Intersection) Score: 0.06194
    Class 9 LB: 0.02113 + LB: 0.02713 (Union) Score: 0.03254

Bad thing, we mostly use leaderboard to check if our ensemble gave boost, which could lead to overfitting. It would be much easier if we have 3rd independent solution to use voting mechanism.

Post processing
---------------
After analysis of class interactions in train I made the following postprocess types:
- Remove too large polygons (from class 8, 9 and 10 for example) 
- Subtract predicted classes polygons like water from car classes

Class 10 with small objects
---------------------------

For class 10 default approach works not very well. The main problems with this class that it’s have very small objects, bad train segmentation and it’s total area is too small comparing to full area. So I created other UNET CNN for this class. It had input of 32x32 pixels. Data for this class was heavily augmented, rotations, different shifts etc. I also used loss function with big penalty for false positives based on [Tversky index][8].

    # https://en.wikipedia.org/wiki/Tversky_index
    def tversky_coef(y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        alfa = 0.1
        false_positive = K.sum(y_pred_f * (1 - y_true_f))
        false_negative = K.sum((1 - y_pred_f) * y_true_f)
        true_positive = K.sum(y_true_f * y_pred_f)
        return (true_positive + 1.0) / (false_negative*alfa + (1 - alfa) * false_positive + true_positive + 1.0)

It allowed me to get 0.00362 score on LB.

Other ideas I tried
-------------------
- At the early beginning I made single XGBoost model which works on squares of size 10x10. Model tries to predict to which class given square is belong. It easily got me higher than baseline. Probably creating independent XGBoost models for each class could give good results. But I switch to CNN without additional experiments with XGBoost approach.
- Pre-trained VGG16 for rare class localization. I tried to predict if car exists in given area with VGG16 on RGB images for later ensemble with UNET predictions.

![Example of class 10 localization with VGG16][9]

- Pre-trained VGG16 for segmentation. The main idea here is that we use final layer (which is now sigmoid instead of softmax) as indication in which part of 224x224 image given class exists. I used 16x16 = 256 neurons for this task. But result wasn’t very good, may be because of low resolution - 14x14 square as single pixel.
- There are big bunch of different indexes used for automatic segmentation without CNN:
http://www.indexdatabase.de/db/i.php
It works great for water as shown in [Waterway Kernel][10]. And these indexes can be added as additional planes during learning process. Unfortunately they didn’t help much in my experiments, so I abandon them. 
- I tried to use CRF (Conditional random field) methods with [pystruct][11] to improve predicted masks. But it somehow won’t work for me.

Independent classes best scores
-------------------------------
![Best Public LB][12]

Segmentation example
--------------------
![enter image description here][13]
![enter image description here][14]

Checkout the video:
**[https://www.youtube.com/watch?v=rpp7ZhGb1IQ][15]**

Code
----

GitHUB link here later. I plan to release code after publication of final results.


  [1]: https://kaggle2.blob.core.windows.net/forum-message-attachments/166431/6037/Tensor_example_700px.png
  [2]: https://kaggle2.blob.core.windows.net/forum-message-attachments/166431/6029/Mask_example.png
  [3]: https://kaggle2.blob.core.windows.net/forum-message-attachments/166431/6038/model_700px.png
  [4]: https://kaggle2.blob.core.windows.net/forum-message-attachments/166431/6031/Training.png
  [5]: https://kaggle2.blob.core.windows.net/forum-message-attachments/166431/6032/LB-Scores.png
  [6]: https://kaggle2.blob.core.windows.net/forum-message-attachments/166431/6033/Sliding%20Window.png
  [7]: https://kaggle2.blob.core.windows.net/forum-message-attachments/166431/6039/Masks_700px.png
  [8]: https://en.wikipedia.org/wiki/Tversky_index
  [9]: https://kaggle2.blob.core.windows.net/forum-message-attachments/166431/6040/vgg16_test_6060_3_2_cls_10_700px.png
  [10]: https://www.kaggle.com/resolut/dstl-satellite-imagery-feature-detection/waterway-0-095-lb
  [11]: https://pystruct.github.io/
  [12]: https://kaggle2.blob.core.windows.net/forum-message-attachments/166431/6036/Best-LB.png
  [13]: https://kaggle2.blob.core.windows.net/forum-message-attachments/166431/6048/6100_1_2_mask.png
  [14]: https://kaggle2.blob.core.windows.net/forum-message-attachments/166431/6049/6100_1_2_proj_700px.jpg
  [15]: https://www.youtube.com/watch?v=rpp7ZhGb1IQ
