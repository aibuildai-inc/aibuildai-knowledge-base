# 9th Place - 2D Detection 3D Classification

Competition: nfl-impact-detection
Rank: #9
Source: https://www.kaggle.com/c/nfl-impact-detection/discussion/209012

# Overview

Our solution (like a lot of other teams) consists of a two-step pipeline (detection and classification) followed by some post-processing. We use 2D detection to find possible impact boxes, and then we use 3D classification to determine which boxes are impacts. 

# Using 2D, Is This a Helmet Impact?

Our detection model finds the yellow box as a possible helmet impact.

From only 2D it is impossible to know whether these helmets are about to impact. It appears probable but maybe the silver helmet is about to pass behind the white helmet.

# Using 3D, Is This a Helmet Impact?

Our classification model uses frames before and after this frame. With this information, our classification model correctly identifies that these two helmets do not impact but rather pass by each other. 


# Impact Detection Model
(written by @jinssaa)

We used DetectoRS(ResNeXt-101-32x4d) model for detecting impacts. https://github.com/open-mmlab/mmdetection This model was built in two steps:

Helmet Detection Model for warm-up: We trained a Helmet Detection Model for 12 epochs using the train helmet dataset which is in the image folder, this model shows 'bbox_mAP_50', 0.877 validation score.
2-Class Detection Model: We used the final weights of the Helmet Detection Model (as pretrained weights) to build a Helmet/Impact Detection model. We benched @tito's notebook and set +- 4 frames from impact as impact class.
Using the Helmet Detection Model as pretrained weight makes the 2-Class Model converge much faster and detect impacts easier.


This approach showed good performance to detect impact. We set the confidence score up to 0.9 which showed 0.39 public LB score after a simple post processing (without any classifiers). Then, we set a lower confidence score to catch more true positives after we plug in the classifier. DetectoRS showed good performance to detect impact but it took a long time to test our ideas.

# Impact Classification Model
(written by @theoviel)

### 2D Models

I kept struggling with improving my EfficientDet models as I had no experience with object detection, I figured out it would be better to go back to what I can do : classification models. The main idea was that a crop around a helmet has the same information regarding whether it is an impact or not as the whole image. 
Therefore, I extracted a 64x64 crop around all the boxes in the training data and started building models to predict whether a crop had an impact. To tackle imbalance, I used the [-4,  +4] extended label as a lot of people did. After a few tweaking, I had a bunch of models with a 0.9+ AUC (5 folds stratified by gameplay) : Resnet-18 & 34, EfficientNet b0 -> b3. 

Tricks used for 2D models include :
Limiting the number of boxes sampled per player at each epoch, in order to have more control on convergence
Linear learning rate scheduling with warmup
Removing the stride of the first layer : 64x64 is a small image size and it’s better not to decrease the network resolution too fast
Stochastic Weighted Averaging
Classical augmentations 

Then, I used @tjungblut’s detector (https://www.kaggle.com/c/nfl-impact-detection/discussion/200995)  to benchmark how bad my models were on the public leaderboard. Turns out that after some post-processing, a resnet-18 + efficientnet-b1 + efficientnet-b3 blend achieved 0.33+, which at that time was in the gold zone.

### Merging

Shortly after, I merged with the rest of the team with the goal of plugging my classification models on top of a strong detection model to leverage their potential. There were about two weeks left before the end of the competition, so we first focused on plugging @Uijin’s detection models with my classifiers. For a while, we couldn’t beat the detector LB’s of 0.39+, but after setting up a good CV scheme and improving the detection model training strategy, we reached LB 0.41+. 

### 3D Models 

Around this time, I upgraded my 2D classifiers to be 3D. Models will now take as input frames [-8, -6, -4, -2, 0, 2, 4, 6, 8] instead of just frame 0. The pipeline was easy to adapt, the data loader was slightly modified, architectures were changed, and augmentations were removed for I was too lazy to dig into that.

The first batch of 3D model is based on https://github.com/kenshohara/3D-ResNets-PyTorch : 
resnet-18 & resnet-34, using as target the same as the one of the middle frame (that was extended)
resnet-18, using an auxiliary classifier to predict the impact type
resnet-18, with a target extended to frames [-6, +6]

The only additional trick I used is getting rid of all the strides on the temporal axis. Models are made for videos longer than 9 frames so once again I adapt the networks to my small input size.

They helped CV, didn’t really help public LB. In fact our jump to 0.49 at the time came from retraining our detection model on the whole dataset & tweaking some post-processing parameters. 
They did however help private LB, but this was after the leak so we didn’t know. 

### More 3D Models

After recovering from NYE, I did some state of the art browsing and implemented 3 new 3D models that all outperformed the previous ones. 
i3d from https://github.com/piergiaj/pytorch-i3d
Slowfast from https://github.com/open-mmlab/mmaction2
Slowonly with Omnisource pretrained weights from https://github.com/open-mmlab/mmaction2

This was done on the 2nd and 3rd of January, so we had 10 submissions to try these out because of my procrastinating. My first submissions using them gave a small LB boost and we reached 0.50+. 

Fortunately the rest of the team worked hard on the detection models to compute results of our detection models on all our folds. This allowed them to find a powerful set of hyperparameters which worked in the end !

# Validation - "Trust Your CV"
(written by @cdeotte)

Building a reliable validation was very important because it was easy to overfit the public LB or a hold out validation set of only a few videos. Our final model was configured using a full 120 video 5 fold validation. By reviewing our OOF predictions, we were able to tell what we needed to do to improve our model. One tool we used was to watch the ground truths and predictions together in all 3 views. More info [here][2] and [here][1]
[[IMAGE ALT TEXT HERE]](http://playagricola.com/Kaggle/labeled_57586_001934-pair_1.mp4)
This gave us insight into choosing hyperparameters for our detection and classification models and it gave us insight into creating post process.

# Post-processing
(written by @vaghefi)

### Thresholding

We had to find good thresholds for both our detection and classification models. We observed that a single split is not enough and F1 score variation between folds is significant. Therefore, we used the CV calculated on the entire train set (5 folds) to optimize our thresholds. The following tricks improved our CV and LB:
Different thresholds for detection and classification model: Originally we had the same threshold for detection and classification models (~0.5). As the classification model became better, we needed to lower detection thresholds (~0.35) to include more helmet detections and then use the classification model to classify the impacts.
Different thresholds for Endzone and Sideline views: Endzone and Sideline videos are different in terms of image content, box area, box size, etc. We realized that using different thresholds can improve both CV and LB. We tried different combinations and best CV was achieved by using around 0.05 higher threshold in Sideline than Endzone
Different thresholds over time: Both detection and classification models have no information about time-elapsed. From the training set, we know that the chance of impact decreases as frame number increases. We tried different combinations (fixed, piecewise, and linear) and ended up using a piecewise method where we used different thresholds for frame >= 150 and frame < 150.



### Box Expansion

Increasing the size of the bounding boxes helped our models, for some reason. In our last submission, we made the bounding boxes 1.22x bigger.

### Adjacency Post-Processing

The goal of this PP was to cluster detections (through multiple frames) which belong to the same player and remove FP. We used an IOU threshold and frame-difference threshold to optimize our algorithm. We also tried temporal NMS and soft-NMS but our custom algorithm performed better. 

### View Post-Processing

Impacts that were detected with a confidence lower than `T` were removed if no impact was found in the other view. That’s the best we came up with regarding merging information from both views.

# Final results

We selected two submissions : Best public LB and best CV.
* CV 0.4885 - **Public 0.5408 (10th)** - Private 0.4873 (13th)
* **CV 0.5125** - Public 0.4931 (13th) - **Private 0.5153 (9th)**. 

Our competition inference notebook is [here][3]

## Takeaways

We should’ve trusted our classification models more, as they were 3D, they most likely were better than our detection model. It would’ve been a better approach to use detection models to detect all helmets instead of only impacts and then use 3D classification models to classify impacts. 

[1]: https://www.kaggle.com/c/nfl-impact-detection/discussion/208782
[2]: https://www.kaggle.com/cdeotte/watch-submission-csv-as-video/
[3]: https://www.kaggle.com/cdeotte/nfl-2d-detect-3d-classify-0-515
